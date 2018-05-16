#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <plplot/plplot.h>

/* Refactored with: https://codebeautify.org/c-formatter-beautifier */

// !!!! non standard library !!!!
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

/* minimum number of arguments:
 *     - argv[0] is the executable name
 *     - argv[1] is the module name */
#define Minarg 2

#define Modules "select_stn, graphic_identify, graphic_select, los_series"

#define Str_IsEqual(string1, string2) (strcmp((string1), (string2)) == 0)
#define Module_Select(string) Str_IsEqual(argv[1], string)

#define errorln(format, ...) fprintf(stderr, format"\n", __VA_ARGS__)

#define C 57.295779513 // 180/pi
#define SYM1 2
#define SYM2 4
#define COLOR 7
#define SCOLOR 3

static PLGraphicsIn gin;

typedef struct {
    char * fn;
}
fname;

typedef struct {
    long int azi, rng, date;
    float lat, lon, hgt, coh;
}
pixcoh;

typedef struct {
    long int azi, rng;
    float lat, lon, hgt, pwr;
}
pixpwr;

typedef struct {
    char * ic;
    float la, fi;
    int pc;
}
cxy;

typedef struct {
    double e, yrd;
    int yr, mn, dy;
}
epoch;

int slc_direction(char * buf)
{
    int i = 0, j, sd = 0;
    
    // change to lower case
    while (*(buf + i)) {
        *(buf + i) = tolower(*(buf + i));
        i++;
    }
    
    j = find_str(buf, "_asc_");
    if (j == 1) sd = 1;
    
    j = find_str(buf, "_dsc_");
    if (j == 1) sd = 2;
    
    return sd;
}

int sign(float val) {
    if (val == 0) return (0);
    else if (val > 0) return (1);
    else if (val < 0) return (-1);
}

void add_phi(int lin, int slin, float * phs, float corr) {
    int i;
    for (i = slin; i < lin; i++)
        *(phs + i) = *(phs + i) + corr;
}

void d_phi(int lin, float * phs, float * dphs) {
    int i;
    for (i = 1; i < lin; i++)
        *(dphs + i) = *(phs + i) - *(phs + i - 1);
}

void unwrap(int lin, int col, float * phase, float trhes) {
    /* Unwrap "col" series of "lin" elements of "phase".
     * Input  "phase" elements are in radian, 
     * Output "phase" elements are LOS values in mm. */

    int i, j, k;
    int sb, sm, sa;

    float * phs, * dphs, dd;
    float wl = 55.465765; // mm

    if ((phs = (float * ) malloc(lin * sizeof(float))) == NULL) {
        printf("\n Not enough memory to allocate phs");
        exit(1);
    }
    if ((dphs = (float * ) malloc(lin * sizeof(float))) == NULL) {
        printf("\n Not enough memory to allocate phs");
        exit(1);
    }

    printf(" ------------------------------------------\n");
    for (j = 0; j < col; j++) // main loop
    {
        // transform radian to cycle and compute differences
        
        // !!!!! 2x / M_PI !!!!!
        for (i = 0; i < lin; i++)
            *(phs + i) = *(phase + i * col + j) / M_PI / M_PI;
        d_phi(lin, phs, dphs);

        //   printf("\n\n IB%d-IB1 cycle\n\n",j+2);
        //   for(k=0;k<lin;k++)
        //   printf(" %2d %8.5f %8.5f\n",k, *(phs+k),  *(dphs+k));
        //   printf("\n----------------------------------------\n\n");

        // search for jumps
        for (i = 1; i < (lin - 1); i++) {
            sb = sign( *(dphs + i - 1));
            sm = sign( *(dphs + i));
            sa = sign( *(dphs + i + 1));

            dd =   abs( *(dphs + i - 1)) / 2.0
                 + abs( *(dphs + i))
                 + abs( *(dphs + i + 1)) / 2.0;

            //     if( sm != sb && sm != sa && dd > trhes )
            if (sm != sb && sm != sa && fabs( * (dphs + i)) > trhes) {
                printf(" %2d %2d %8.5f %8.5f %8.5f  %8.5f\n",
                    j, i, *(dphs + i - 1), *(dphs + i), *(dphs + i + 1), dd);

                if (sm == 1)
                    add_phi(lin, i, phs, -1.0);
                else
                    add_phi(lin, i, phs, 1.0);
                d_phi(lin, phs, dphs);

                //   printf("\n\n IB%d-IB1 %d\n\n",j+2,i);
                //   for(k=0;k<lin;k++)
                //   printf(" %2d %8.5f %8.5f\n",k, *(phs+k),  *(dphs+k));
                //   printf("\n");
            } // end if
        }
        // end search for jumps
        printf(" ------------------------------------------\n");

        for (k = 0; k < lin; k++)
            *(phase + k * col + j) = - *(phs + k) / 2.0 * wl;
    } // end  main loop

    for (i = 0; i < lin; i++) {
        printf("\n %2d ", i);
        for (j = 0; j < col; j++)
            printf(" %8.2f", *(phase + i * col + j));
    }
    printf("\n\n ------------------------------\n\n");

    free(phs);

} // end unwrap

void swap_ibs(int nib, int nr, cxy * ibsd) {
    // referebce IB swap position with last IB
    char * buf;
    int pc;
    double la, fi;

    if ((buf = (char * ) malloc(4 * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate BUF");
        exit(1);
    }

    strcpy(buf, (ibsd + nr)->ic);
    la = (ibsd + nr)->la;
    fi = (ibsd + nr)->fi;
    pc = (ibsd + nr)->pc;

    strcpy((ibsd + nr)->ic, (ibsd + nib - 1)->ic);
    (ibsd + nr)->la = (ibsd + nib - 1)->la;
    (ibsd + nr)->fi = (ibsd + nib - 1)->fi;
    (ibsd + nr)->pc = (ibsd + nib - 1)->pc;

    strcpy((ibsd + nib - 1)->ic, buf);
    (ibsd + nib - 1)->la = la;
    (ibsd + nib - 1)->fi = fi;
    (ibsd + nib - 1)->pc = pc;
} // end swap_ibs

void master_date(char * buf, char * date) {
    /* subtract master date from full path and
     * store in double precision (yyymmdd.0)
     * similarly to "date.txt" file */

    int i, n;
    char m[7];
    char d[9];

    n = strlen(buf);

    if ( *(buf + n - 1) != '/') {
        printf("\n\n Missing '/' character at the end of full path !\n\n");
        exit(1);
    }

    n = n - 14;
    m[6] = '\0';

    do {
        n = n - 1;
        for (i = 0; i < 6; i++)
            m[i] = *(buf + n + i);

    } while (strcmp( &m[0], "INSAR_") != 0);

    for (i = 0; i < 8; i++)
        date[i] = *(buf + i + n + 6);
    date[8] = '\0';

} // end master_date

double master_date_los(char * buf) {
    /* subtract master date from full path and
     * store in double precision (yyymmdd.0)
     * similarly to "date.txt" file */

    int i, n;
    char m[7];
    char d[9];

    n = strlen(buf);

    if ( *(buf + n - 1) != '/') {
        printf("\n\n Missing '/' character at the end of full path !\n\n");
        exit(1);
    }

    n = n - 14;
    m[6] = '\0';

    do {
        n = n - 1;
        for (i = 0; i < 6; i++)
            m[i] = *(buf + n + i);

    } while (strcmp( &m[0], "INSAR_") != 0);

    for (i = 0; i < 8; i++)
        d[i] = *(buf + i + n + 6);
    d[8] = '\0';

    return (strtof(d, NULL));

} // end master_date

void decimal_year(int n, epoch * meas, double h) {
    // Compute the decimal years starting from initial year.

    int i, j, year;
    unsigned long days; // day counter from initial year

    //                       1   2   3   4   5    6    7    8    9    10   11   12
    int day_count[2][12] = {
        { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 },
        { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335 }
    };
    // initial epoch

    days = 0;
    year = (meas + 0)->yr;
    i = 0;
    if (((meas + 0)->yr % 4) == 0)
        i = 1;
    (meas + 0)->yrd +=   ((days + day_count[i][(meas + 0)->mn - 1]
                       + (meas + 0)->dy - 1) * 1.0 + h / 24.0) / 365.25;
    
    // other epochs
    for (j = 1; j < n; j++) {
        while ((meas + j)->yr > year) {
            year++;
            if ((year % 4) == 0)
                days += 366;
            else
                days += 365;
        }
        i = 0;
        if (((meas + i)->yr % 4) == 0)
            i = 1;
        (meas + j)->yrd = (meas + 0)->yr
                          + ((days + day_count[i][(meas + j)->mn - 1]
                          + (meas + j)->dy - 1) * 1.0 + h / 24.0) / 365.25;
    }
} // end decimal_year

// -------------------------------------------------------------------
void suspend(int i) {
    printf("\n\n suspend(%d) => Ctrl c to exit - Enter to continue !\n", i);
    getchar();
} // end suspend

double hms_rad(char * hms) {
    // return angle in radian subtracted 
    // from formatted string (h-m-s.sss) 
    int h, m;
    double s;
    sscanf(hms, "%d-%d-%lf", &h, &m, &s);
    return ((h * 1.0 + m / 60.0 + s / 3600.0) / 180.0 * M_PI);
} //end hms_rad

void def_rectangle(FILE * stn, float suf, float * lamn, float * lamx,
    float * fimn, float * fimx) {
    float fi, la, fim, lam, he;
    char shid[4],  // IB short identifier   -  "IB1"   
         name[21], // name of IB            -  "Deak_str"
         lats[20], // latitude  of IB       -  "47-12-43.12345"
         lons[20]; // longitude of IB       -  "16-14-21.12345" 
    int id;

    *lamn = *fimn = 360.0;
    *lamx = *fimx = lam = fim = 0.0;

    while (fscanf(stn, "%s %s %s %s %f %d",
                    shid, name, lats, lons, &he, &id) > 0) {
        fi = hms_rad(lats) * C;
        la = hms_rad(lons) * C;

        //    fprintf(out,"%9.6f %9.6f %5.1f  %s\n",la,fi,he,shid);

        if (la < * lamn)
            *lamn = la;
        if (la > * lamx)
            *lamx = la;
        if (fi < * fimn)
            *fimn = fi;
        if (fi > * fimx)
            *fimx = fi;
    }
    lam = ( *lamn + *lamx) / 2.0;
    fim = ( *fimn + *fimx) / 2.0;
    fi = *fimx - fim;
    la = *lamx - lam;
    *lamn = lam - la * suf;
    *lamx = lam + la * suf;
    *fimn = fim - fi * suf;
    *fimx = fim + fi * suf;

    //    fprintf(out,"\n");
}
// end def_rectangle

// -------------------------------
void sort_coh(pixcoh * a, int n) {
    // increasing order
    int i, j;
    pixcoh temp;

    for (i = 0; i < (n - 1); i++)
        for (j = 0; j < (n - 1 - i); j++)
            if (a[j].coh > a[j + 1].coh) {
                temp = a[j + 1];
                a[j + 1] = a[j];
                a[j] = temp;
            }
} // end sort_coh

void sort_azi_cc(pixcoh * a, int n) {
    // increasing azimuth (azi) order
    int i, j;
    pixcoh temp;

    for (i = 0; i < (n - 1); i++)
        for (j = 0; j < (n - 1 - i); j++)
            if (a[j].azi > a[j + 1].azi) {
                temp = a[j + 1];
                a[j + 1] = a[j];
                a[j] = temp;
            }
} // end sort_azi_cc

void sort_rng_cc(pixcoh * a, int b, int e) {
    // increasing range (rng) order
    int i, j;
    pixcoh temp;

    for (i = 0; i < (e - b); i++)
        for (j = 0; j < (e - b - i); j++)
            if (a[j + b - 1].rng > a[j + b].rng) {
                temp = a[j + b];
                a[j + b] = a[j + b - 1];
                a[j + b - 1] = temp;
            }
} // end sort_rng_cc

void sort_azi_pwr(pixpwr * a, int n) {
    // increasing azimuth (azi) order
    int i, j;
    pixpwr temp;

    for (i = 0; i < (n - 1); i++)
        for (j = 0; j < (n - 1 - i); j++)
            if (a[j].azi > a[j + 1].azi) {
                temp = a[j + 1];
                a[j + 1] = a[j];
                a[j] = temp;
            }
} // end sort_azi_pwr

void sort_rng_pwr(pixpwr * a, int b, int e) {
    // increasing range (rng) order
    int i, j;
    pixpwr temp;

    for (i = 0; i < (e - b); i++)
        for (j = 0; j < (e - b - i); j++)
            if (a[j + b - 1].rng > a[j + b].rng) {
                temp = a[j + b];
                a[j + b] = a[j + b - 1];
                a[j + b - 1] = temp;
            }
} // end sort_rng_pwr

void fswap(float * f) {
    char * in = (char * ) f;
    float u;
    char * ou = (char * ) & u;
    ou[0] = in [3];
    ou[1] = in [2];
    ou[2] = in [1];
    ou[3] = in [0]; * f = u;
} // end fswap

//-----------------------------------
int find_str(char * buf, char * str) {
    //  find "str" string in buf
    //  return 1 if found else 0

    int i, j, k;
    int r = 0;
    j = strlen(str);
    k = strlen(buf);

    for (i = 0; i < (k - j + 1); i++)
        if (strncmp((buf + i), str, j) == 0)
            r = 1;

    return (r);
} // end find_str 

// -------------------------------------------
void change_f_ext(char * name, char * ext) {
    // change the extent of name for ext

    int i = 0;
    while ( *(name + i) != '.' && *(name + i) != '\0') i++;
    *(name + i) = '\0';

    sprintf(name, "%s%s", name, ext);

} // end change_f_ext

void selplot(PLFLT * x, PLFLT * y, int is, int sym, int color) {
    // plots the selected point

    plcol0(color);
    plpoin(1, x + is, y + is, sym);
}

int psearch(PLINT np, PLFLT * x, PLFLT * y, double dd, int nsel, int * sel) {
    int j, k, is;
    double ddd;

    for (j = 0; j < np; j++) {
        ddd = sqrt(pow(x[j] - gin.wX, 2) + pow(y[j] - gin.wY, 2));

        if (ddd < dd) {
            for (k = 0; k < nsel; k++) {
                if (sel[k] == j) {
                    printf(" double choice\n");
                    return (-1);
                }
            }
            is = j;
            return (is);
        }
    }

    is = -1;
    return (is);

} // end psearch

// ------------------------
int max_reclen(FILE * inp) {
    // count the max. line length
    // for memory allocation

    char c;
    int max = 0,
          n = 0;

    while ((c = getc(inp)) != EOF) {
        if (c == 0x0a && n > max)
            max = n;
        else if (c == 0x0a)
            n = 0;
        else
            n++;
    }
    return (max + 2);

} // end max_reclen

void change_m_ext(char * name, char * ext) {
    // change the extent of name for ext

    int i = 0;
    while ( *(name + i) != '.' && *(name + i) != '\0') i++;
    *(name + i) = '\0';
    sprintf(name, "%s%s", name, ext);
} // end change_m_ext


/****************
* MAIN MODULES *
****************/

int cc_thinning(int argc, char * argv[]) {

    char cwd[1024];

    DIR * data;
    struct dirent * inp; // dir pointer: inp->d_reclen, inp->d_name

    FILE *lat, *lon, // geometric file id-s
               *stn, // station   file id
         *cci, *cco, // input and output coh file id-s
               *par, // mdate mli.par file
               *log;

    long int ii, jj,   // long indeces
             na, np,   // long indeces
             lin, col; // lines and coloumns of data files
    //           date;     

    int i, j, b, e, n;
    int thm;

    fname * infs; // names of rmli files
    pixcoh * sub, * all;

    int no; // number of cc files

    char * files;
    char * ddir, * c;

    char mdate[9];

    float suf;
    float lamn, lamx, fimn, fimx;
    float latc, lonc, ccid;

    /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     *   0          1      2       3        4     5   6    7
     * isign cc_thinning fullpath mdate station.stn scale thm
     *++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

    // --------------------------------------------------------------------
    if ((files = (char * ) malloc(1024 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate files\n\n");
        exit(1);
    }
    if ((ddir = (char * ) malloc(1024 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate ddir\n\n");
        exit(1);
    }

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                       CC_THINNING                          +\
            \n +   ~60 pc thinning or copy the *.cc_ad files into *.cc      +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    if (argc - Minarg < 3) {
        printf("\n  usage:                                                       \
                \n          isign cc_thinning /full_path/ network.stn 1.5 1      \
                \n  /full_path/ - full_path of GAMMA preproc directory           \
                \n                (or . in GAMMA preproc directory)              \
                \n  network.stn - IBs network file copied into working directory \
                \n          1.5 - scale-up factor of network search rectangle    \
                \n           1  - 1 thinning, 0 only copy                        \
                \n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");
        return (1);
    }

    if ((log = fopen("cc_thinning.log", "wt")) == NULL) {
        printf("\n\n Cannot create LOG file !\n\n");
        exit(1);
    }

    for (i = 0; i < argc; i++)
        fprintf(log, " %s", argv[i]);

    // printf("\n\n");
    // for(i=0;i<argc;i++)
    // printf("%s\n",argv[i]);

    c = argv[1];
    if ( * c == '.') {
        getcwd(cwd, sizeof(cwd));
        sprintf(ddir, "%s%s", cwd, "/");
    } else ddir = argv[1];

    if (find_str(ddir, "INSAR_") == 0) {
        printf("\n ERROR: INSAR_yyyymmdd is not included in directory name\n\n");
        exit(1);
    }

    master_date(argv[1], mdate); // master date

    sprintf(files, "%s%s.mli.par", ddir, mdate);

    printf(" master date: %s\n\n", mdate);
    fprintf(log, "\n\n master date: %s\n\n", mdate);

    if ((par = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    while (fgets(cwd, 1024, par) > 0) {
        if (find_str(cwd, "range_samples:") == 1)
            sscanf(cwd, "%s %ld", files, & col);
        if (find_str(cwd, "azimuth_lines:") == 1)
            sscanf(cwd, "%s %ld", files, & lin);
    }

    printf(" lines:  %6ld \n columns: %ld   of Gamma files\n", lin, col);
    fprintf(log, " lines:  %6ld \n columns: %ld   of Gamma files\n", lin, col);

    sscanf(argv[4], "%f", & suf); // scale-up
    printf("\n scale-up: %6.2f\n", suf);
    fprintf(log, "\n scale-up: %6.2f\n", suf);

    sscanf(argv[5], "%d", & thm); // 1 - thinning,0 - only copy

    if (thm == 1) {
        printf(" solution: %3d    - thinning\n", thm);
        fprintf(log, "\n solution: %3d    - thinning\n", thm);
    } else {
        thm = 0;
        printf(" solution: %3d    - only copy\n", thm);
        fprintf(log, "\n solution: %3d    - only copy\n", thm);
    }

    sprintf(files, "%s", argv[3]); // stn file
    if ((stn = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }
    def_rectangle(stn, suf, & lamn, & lamx, & fimn, & fimx);
    printf("\n search area:  %9.6f %9.6f\n               %9.6f %9.6f\n\n",
        lamn, lamx, fimn, fimx);
    fprintf(log, "\n search area:  %9.6f %9.6f\n               %9.6f %9.6f\n\n",
        lamn, lamx, fimn, fimx);

    printf(" input files:\n\n");
    fprintf(log, " input files:\n\n");

    printf(" %s\n\n", files);
    fprintf(log, " %s\n\n", files);

    sprintf(files, "%s%s.lat", ddir, mdate); // lat file
    if ((lat = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((lat = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s%s.lon", ddir, mdate); // lon file
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((lon = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s%s.mli.par", ddir, mdate);
    printf(" %s\n", files); // proc_par file
    fprintf(log, " %s\n", files);

    // read cc_ad file names from dir data

    if ((infs = (fname * ) malloc(1 * sizeof(fname))) == NULL) {
        printf("\n\n Not enough memory to allocate infs\n\n");
        exit(1);
    }
    if ((infs->fn = (char * ) malloc(14 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate infs->fn\n\n");
        exit(1);
    }

    no = 0;
    data = opendir(ddir);
    while (inp = readdir(data)) {
        //   printf(" %d %s\n",inp->d_reclen,inp->d_name );
        if ((inp->d_reclen == 48) && (find_str(inp->d_name, ".cc_ad") == 1)) {
            strcpy((infs + no)->fn, inp->d_name);
            no++;
            if ((infs = (fname * ) realloc(infs, (no + 1) * sizeof(fname))) == NULL) {
                printf("\n\n Not enough memory to reallocate infs\n\n");
                exit(1);
            }
            if (((infs + no)->fn = (char * ) malloc(14 * sizeof(char))) == NULL) {
                printf("\n\n Not enough memory to allocate infs->fn\n\n");
                exit(1);
            }
        }
    }
    closedir(data);

    sprintf(files, "%s%s", ddir, "yyyyddmm_yyyyddmm.cc_ad");
    printf("\n %s", files);
    fprintf(log, "\n %s", files);

    printf("\n\n output files:\n\n");
    fprintf(log, "\n\n output files:\n\n");

    printf(" cc_thinning.log\n\n");
    fprintf(log, " cc_thinning.log\n\n");

    //  end read cc file names from dir data
    //-------------------------------------------------------------------
    //--------------------------------------------------------------------
    // select all candidates

    for (i = 0; i < no; i++) {
        np = 1;
        na = 0;

        sprintf(files, "%s%s", ddir, (infs + i)->fn);
        if ((cci = fopen(files, "rb")) == NULL) {
            printf("\n %s data file not found ! \n\n", files);
            exit(1);
        }

        change_f_ext(files, ".cc");
        if ((cco = fopen(files, "wb")) == NULL) {
            printf("\n %s data file not found ! \n\n", files);
            exit(1);
        }

        change_f_ext((infs + i)->fn, ".cc");
        printf(" %s", (infs + i)->fn);
        fprintf(log, " %s", (infs + i)->fn);

        for (ii = 0; ii < lin; ii++)
            for (jj = 0; jj < col; jj++) {
                fread( &latc, sizeof(float), 1, lat);
                fswap( &latc);
                fread( &lonc, sizeof(float), 1, lon);
                fswap( &lonc);
                fread( &ccid, sizeof(float), 1, cci);

                if (latc > fimn && latc < fimx && lonc > lamn && lonc < lamx) na++;
                else {
                    if (thm == 1 && ((np % 2) == 0 || (np % 5) == 0 || (np % 9) == 0)) ccid = 0.0;
                    else na++;
                }
                fwrite( & ccid, sizeof(float), 1, cco);
                np++;

            } // end for "lin" "col"

        fclose(cci);
        fclose(cco);
        rewind(lat);
        rewind(lon);

        printf("  ***  %6ld  of  %ld *.cc_ad are reserved\n", na, np - 1);
        fprintf(log, "  ***  %6ld  of  %ld *.cc_ad are reserved\n", na, np - 1);

    } // end for "no"

    printf("\n\n");

    fclose(lat);
    fclose(lon);
    fclose(stn);
    fclose(par);

    return (1);

} // end cc_thinning

/******************************************************************************
*                           GRAPHIC_CONTROL                                  *
*                                                                            *
* preselected PSs may be deleted or identified as known IBs                  *
*                                                                            *
* In the case of 0 option only PSs can be deleted.                           *
* In the case of 1 option PSs and IBs can be coupled as identified pair      *
*                                                                            *
* 0 : left click  on 'o' change the colour to green and delete the PS        *
*                                                                            *
* 1 : left clicks on 'o' (PS) then on '+' (IB) change the colour to green    *
*     and couple the pairs, midle click on map to cancel last pair           *
*                                                                            *
*     to quit : double right click on map                                    *
*****************************************************************************/

int graphic_control(int argc, char * argv[]) {

    PLFLT *x, *y;
    PLFLT *x2, *y2;

    PLFLT xmin, xmax, ymin, ymax;
    PLINT np1, // number of PSs
          np2; // number of IBs

    int np, // number of PSs
        nb; // number of IBs

    double dd;
    int is, i1, i2;

    char * chh;
    int * hi;

    int j, nsel, option;

    int *sel1, *sel2, *seld;

    FILE *inp, // input PSs
         *inb, // input IBs
         *tem, // temporary PSs or IBs station.sti
         *log; // log file

    double xx, yy;
    char *buf; // buffer
    int l, // max length of input records
        nd, // number of deleted PSs
        pc; // IB proc code

    char sid[4], // short id of IBs
         lid[21]; // long  id of IBs

    int lad, lam;
    float las;
    int fid, fim;
    float fis,
    hei;

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                 GRAPHIC_CONTROL                    +\
            \n +   selected PSs and identified IBs are controlled   +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    if (argc - Minarg < 2) {
        printf("\n  usage:  isign graphic_control data.xys network.sti   \
                \n       data.xys  - (asc or dsc) selected data file     \
                \n    network.sti  - (*.sta or *.std) identified IBs     \
                \n        to quit  - double right click on map           \
                \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

        exit(1);
    }

    sscanf(argv[3], "%d", & option);

    option = 0; //new

    if (option == 0) {
        if ((log = fopen("graphic_0.log", "wt")) == NULL) {
            printf("\n\n Cannot creae LOG file !\n\n");
            exit(1);
        }
    } else {
        if ((log = fopen("graphic_1.log", "wt")) == NULL) {
            printf("\n\n Cannot creae LOG file !\n\n");
            exit(1);
        }
    }

    fprintf(log, "\n ");
    for (j = 0; j < argc; j++)
        fprintf(log, " %s", argv[j]);

    printf("\n input PSs data file: %s", argv[1]);
    printf("\n input IBs data file: %s\n", argv[2]);

    fprintf(log, "\n\n input PSs data file: %s", argv[1]);
    fprintf(log, "\n input IBs data file: %s\n", argv[2]);

    if ((inp = fopen(argv[1], "rt")) == NULL) {
        printf("\n\n %s Input file not found !\n\n", argv[1]);
        exit(1);
    }

    if ((inb = fopen(argv[2], "rt")) == NULL) {
        printf("\n\n %s Input file not found !\n\n", argv[1]);
        exit(1);
    }

    l = max_reclen(inp);
    rewind(inp);

    j = max_reclen(inb);
    rewind(inb);

    if (j > l) l = j;

    if ((buf = (char * ) malloc(l * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate BUF");
        exit(1);
    }

    if (option == 0) {
        strcpy(buf, "tempfile.xys");

        if ((tem = fopen(buf, "wt")) == NULL) {
            printf("\n\n Cannot create TEMPFILE !\n\n");
            exit(1);
        }

        //    printf("\n modified input file: %s\n",argv[1]);
        //    fprintf(log,"\n modified input file: %s\n",argv[1]);
    } else {
        strcpy(buf, argv[2]);
        if (find_str(argv[1], "asc") == 1)
            change_m_ext(buf, "_asc.sti");
        else
        if (find_str(argv[1], "dsc") == 1)
            change_m_ext(buf, "_dsc.sti");
        else change_m_ext(buf, "_!sc.sti");

        //    if( (tem = fopen(buf,"wt")) == NULL )
        //   { printf("\n\n Cannot create TEMPFILE !\n\n"); exit(1); } 

        printf("\n identified IBs file: %s\n", buf);
        fprintf(log, "\n identified IBs file: %s\n", buf);
    }

    np = 0;
    while (fgets(buf, l, inp) != NULL) np++;
    rewind(inp); // number of PSs
    np1 = np; // number in PLINT

    printf("\n number of PSs: %2d\n", np);
    fprintf(log, "\n number of PSs: %2d", np);

    if ((chh = (char * ) malloc(np * 4 * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate CHH");
        exit(1);
    }
    if ((hi = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate HI");
        exit(1);
    }
    if ((sel1 = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SEL1");
        exit(1);
    }
    if ((sel2 = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SEL2");
        exit(1);
    }
    if ((seld = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SELD");
        exit(1);
    }

    if ((x = (PLFLT * ) malloc(np * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate X");
        exit(1);
    }
    if ((y = (PLFLT * ) malloc(np * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate Y");
        exit(1);
    }

    // read PSs
    for (j = 0; j < np; j++) {
        fgets(buf, l, inp);
        sscanf(buf, "%lf%lf", & xx, & yy);
        *(x + j) = xx; * (y + j) = yy;
    }
    for (j = 0; j < np; j++) seld[j] = 0;
    // end read PSs

    printf("\n PSs      longitude    latitude\n\n");
    for (j = 0; j < np; j++)
        printf(" %3d     %11.6f %11.6f\n", j + 1, x[j], y[j]);

    nb = 0;
    while (fgets(buf, l, inb) != NULL) nb++;
    rewind(inb); // number of IBs
    np2 = nb; // number in PLINT

    printf("\n number of identified IBs: %2d\n", nb);
    fprintf(log, "\n number of identified IBs: %2d", nb);

    if ((x2 = (PLFLT * ) malloc(nb * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate X2");
        exit(1);
    }
    if ((y2 = (PLFLT * ) malloc(nb * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate Y2");
        exit(1);
    }

    // read IBs
    for (j = 0; j < nb; j++) {
        fgets(buf, l, inb);

        //   sscanf(buf,"%s %s %d-%d-%f %d-%d-%f %f %d",sid,lid, &lad,&lam,&las, &fid,&fim,&fis,&hei, &pc);

        sscanf(buf, "%s %f %f %d", sid, & las, & fis, & pc);

        strcpy(chh + j * 4, sid); * (chh + j * 4 + 3) = 0;

        //   *(y2+j)= lad*1.0+lam/60.0+las/3600.0;
        //   *(x2+j)= fid*1.0+fim/60.0+fis/3600.0;

        * (y2 + j) = fis; * (x2 + j) = las; * (hi + j) = pc;
    }

    printf("\n IBs      longitude    latitude\n");
    for (j = 0; j < nb; j++)
        printf("\n %2d %s %11.6f %11.6f %d",
            j + 1, (chh + j * 4), * (x2 + j), * (y2 + j), * (hi + j));

    xmin = 1.0e20;
    xmax = -1.0e20;
    for (j = 0; j < np1; j++) {
        xmin = (xmin < x[j]) ? xmin : x[j];
        xmax = (xmax > x[j]) ? xmax : x[j];
    }

    xmin = xmin - (xmax - xmin) * 0.3;
    xmax = xmax + (xmax - xmin) * 0.3;
    
    ymin = 1.0e20;
    ymax = -1.0e20;
    
    for (j = 0; j < np1; j++) {
        ymin = (ymin < y[j]) ? ymin : y[j];
        ymax = (ymax > y[j]) ? ymax : y[j];
    }

    ymin = ymin - (ymax - ymin) * 0.3;
    ymax = ymax + (ymax - ymin) * 0.3;
    dd = (xmax - xmin) * 0.01;
    
    plsdev("xwin");
    plspage(100.0, 100.0, 1000, 750, 30, 30);
    
    plinit();
    
    plenv(xmin, xmax, ymin, ymax, 0, 0);
    plcol0(COLOR);
    
    pllab("longitude", "latitude", "PSs o IBs +");
    plpoin(np1, x, y, SYM2);
    plpoin(np2, x2, y2, SYM1);
    nsel = 0;

    do {
        plGetCursor( & gin);

        // delete the last pair
        if (gin.button == 2 && option == 1) {
            if (nsel > 0) {
                nsel--;
                is = sel1[nsel];
                selplot(x, y, is, SYM2, COLOR);
                is = sel2[nsel];
                selplot(x2, y2, is, SYM1, COLOR);
            }
            continue;
        }
        if (gin.button == 1) {
            do {
                while (gin.button != 1) plGetCursor( & gin);
                is = psearch(np1, x, y, dd, nsel, sel1);
                if (is == -1) plGetCursor( & gin);

            } while (is == -1);

            sel1[nsel] = is;
            seld[is] = 1;
            selplot(x, y, is, SYM2, SCOLOR);

            //    coupling    
            if (option == 1) {
                gin.button = 0;
                //      printf("1: choice point from the 2nd set (stn,+)\n ");
                do {
                    while (gin.button != 1) plGetCursor( & gin);
                    is = psearch(np2, x2, y2, dd, nsel, sel2);
                    if (is == -1) plGetCursor( & gin);

                } while (is == -1);

                sel2[nsel] = is;
                nsel++;
                selplot(x2, y2, is, SYM1, SCOLOR);
            }
        }
    } while (gin.button != 3);

    if (option == 0) {
        rewind(inp);
        nd = np;
        //   printf("\n\n                 modified \n PSs      longitude    latitude\n\n");

        for (j = 0; j < np; j++) {
            fgets(buf, l, inp);
            if (seld[j] == 0) {
                //      printf(" %2d     %11.6f %11.6f\n",j+1, x[j], y[j]);      
                //      fprintf (tem,"%s",buf);
                nd--;
            }
        }

        //     strcpy(buf,"tempfile.xys");
        //     remove(argv[1]);
        //     rename(buf,argv[1]);

        //     printf("\n deleted   PSs: %2d\n",nd);
        //     fprintf(log,"\n deleted   PSs: %2d",nd);

    } // end if

    if (option == 1) {
        for (j = 0; j < nsel; j++) {
            i1 = sel1[j];
            i2 = sel2[j];
            x2[i2] = x[i1];
            y2[i2] = y[i1];
        }

        printf("\n\n                identified\n IBs      longitude    latitude\n");
        for (j = 0; j < np2; j++) {
            printf("\n %2d %s %11.6f %11.6f %d", j + 1, (chh + j * 4),
                                                 x2[j], y2[j], hi[j]);
            fprintf(tem, "%s %11.6f %11.6f %d\n", (chh + j * 4),
                                                  x2[j], y2[j], hi[j]);
        }
    }

    plend();

    printf("\n\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                END GRAPHIC_CONTROL                   +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    exit(0);

} // end graphic_identify

/******************************************************************************
*                            GRAPHIC_IDENTIFY                                *
*                                                                            *
* preselected PSs may be deleted or identified as known IBs                  *
*                                                                            *
* In the case of 0 option only PSs can be deleted.                           *
* In the case of 1 option PSs and IBs can be coupled as identified pair      *
*                                                                            *
* 0 : left click  on 'o' change the colour to green and delete the PS        *
*                                                                            *
* 1 : left clicks on 'o' (PS) then on '+' (IB) change the colour to green    *
*     and couple the pairs, midle click on map to cancel last pair           *
*                                                                            *
*     to quit : double right click on map                                    *
*****************************************************************************/

int graphic_identify(int argc, char * argv[]) {

    PLFLT * x, * y;
    PLFLT * x2, * y2;

    PLFLT xmin, xmax, ymin, ymax;
    PLINT np1, // number of PSs
          np2; // number of IBs

    int np, // number of PSs
        nb; // number of IBs

    double dd;
    int is, i1, i2;

    char *chh;
    int *hi;

    int j, nsel, option;

    int *sel1, *sel2, *seld;

    FILE *inp, // input PSs
         *inb, // input IBs
         *tem, // temporary PSs or IBs station.sti
         *log; // log file

    double xx, yy;
    char *buf; // buffer
    int l, // max length of input records
        nd, // number of deleted PSs
        pc; // IB proc code

    char sid[4], // short id of IBs
         lid[21]; // long  id of IBs

    int lad, lam;
    float las;
    int fid, fim;
    float fis,
    hei;

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                GRAPHIC_IDENTIFY                    +\
            \n +    preselected PSs are identified as known IBs     +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    if (argc - Minarg < 3) {
        printf("\n  usage:  isign graphic_identify data.xys network.stn 0 \n\
                \n      data.xys  - (ASC or DSC) selected data file      \
                \n   network.stn  - (*.stn) IB network to identify       \
                \n                - (*.sta or *std) to control identity  \
                \n             0  - 0 -> delete PSs or control identity  \
                \n                - 1 -> couple PS and IB pair (*.stn)   \n\
                \n         0 : left click  on 'o'(PS)                    \
                \n         1 : left clicks on 'o'(PS) then on '+'(IB)    \n\
                \n             midle click on map to cancel last pair    \n\
                \n   to quit : double right click on map                 \
                \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

        exit(1);
    }

    sscanf(argv[3], "%d", & option);

    if (find_str(argv[2], ".stn") != 1) option = 0;

    if ((log = fopen("ib_graphic_identify.log", "wt")) == NULL) {
        printf("\n\n Cannot creae LOG file !\n\n");
        exit(1);
    }

    fprintf(log, "\n ");
    for (j = 0; j < argc; j++) fprintf(log, " %s", argv[j]);

    printf("\n input PSs data file: %s", argv[1]);
    printf("\n input IBs data file: %s\n", argv[2]);

    fprintf(log, "\n\n input PSs data file: %s", argv[1]);
    fprintf(log, "\n input IBs data file: %s\n", argv[2]);

    if ((inp = fopen(argv[1], "rt")) == NULL) {
        printf("\n\n %s Input file not found !\n\n", argv[1]);
        exit(1);
    }

    if ((inb = fopen(argv[2], "rt")) == NULL) {
        printf("\n\n %s Input file not found !\n\n", argv[2]);
        exit(1);
    }

    l = max_reclen(inp);
    rewind(inp);

    j = max_reclen(inb);
    rewind(inb);

    if (j > l) l = j;

    if ((buf = (char * ) malloc(l * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate BUF");
        exit(1);
    }

    if (option == 0) {
        strcpy(buf, "tempfile.xys");

        if ((tem = fopen(buf, "wt")) == NULL) {
            printf("\n\n Cannot create TEMPFILE !\n\n");
            exit(1);
        }

        if (find_str(argv[2], ".stn") == 1) {
            printf("\n modified input file: %s\n", argv[1]);
            fprintf(log, "\n modified input file: %s\n", argv[1]);
        }
    } else if (option == 1) {
        strcpy(buf, argv[2]);
        if (find_str(argv[1], "asc") == 1)
            change_m_ext(buf, ".sta");
        else if (find_str(argv[1], "dsc") == 1)
            change_m_ext(buf, ".std");
        else
            change_m_ext(buf, ".ngv");

        if ((tem = fopen(buf, "wt")) == NULL) {
            printf("\n\n Cannot create TEMPFILE !\n\n");
            exit(1);
        }

        printf("\n identified IBs file: %s\n", buf);
        fprintf(log, "\n identified IBs file: %s\n", buf);
    }

    np = 0;
    while (fgets(buf, l, inp) != NULL) np++;
    rewind(inp); // number of PSs
    np1 = np; // number in PLINT

    printf("\n number of PSs: %2d\n", np);
    fprintf(log, "\n number of PSs: %2d", np);

    if ((chh = (char * ) malloc(np * 4 * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate CHH");
        exit(1);
    }
    if ((hi = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate HI");
        exit(1);
    }
    if ((sel1 = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SEL1");
        exit(1);
    }
    if ((sel2 = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SEL2");
        exit(1);
    }
    if ((seld = (int * ) malloc(np * sizeof(int))) == NULL) {
        printf("\n Not enough memory to allocate SELD");
        exit(1);
    }

    if ((x = (PLFLT * ) malloc(np * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate X");
        exit(1);
    }
    if ((y = (PLFLT * ) malloc(np * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate Y");
        exit(1);
    }

    // read PSs
    for (j = 0; j < np; j++) {
        fgets(buf, l, inp);
        sscanf(buf, "%lf%lf", & xx, & yy); * (x + j) = xx; * (y + j) = yy;
    }
    for (j = 0; j < np; j++) seld[j] = 0;
    // end: read PSs

    printf("\n PSs      longitude    latitude\n\n");
    for (j = 0; j < np; j++) printf(" %2d     %11.6f %11.6f\n", j + 1, x[j], y[j]);
    nb = 0;
    while (fgets(buf, l, inb) != NULL) nb++;
    rewind(inb); // number of IBs
    np2 = nb; // number in PLINT

    printf("\n number of IBs: %2d\n", nb);
    fprintf(log, "\n number of IBs: %2d", nb);

    if ((x2 = (PLFLT * ) malloc(nb * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate X2");
        exit(1);
    }
    if ((y2 = (PLFLT * ) malloc(nb * sizeof(PLFLT))) == NULL) {
        printf("\n Not enough memory to allocate Y2");
        exit(1);
    }

    // read IBs
    for (j = 0; j < nb; j++) {
        fgets(buf, l, inb);

        if (find_str(argv[2], ".stn") == 1) {
            sscanf(buf, "%s %s %d-%d-%f %d-%d-%f %f %d", sid, lid, & lad,
                                                         &lam, &las, &fid,
                                                         &fim, & fis, &hei, &pc);
            strcpy(chh + j * 4, sid);
            *(chh + j * 4 + 3) = 0;
            *(y2 + j) = lad * 1.0 + lam / 60.0 + las / 3600.0;
            *(x2 + j) = fid * 1.0 + fim / 60.0 + fis / 3600.0; * (hi + j) = pc;
        } else {
            sscanf(buf, "%s %f %f %d", sid, & las, & fis, & pc);
            strcpy(chh + j * 4, sid);
            *(chh + j * 4 + 3) = 0;
            *(y2 + j) = fis;
            *(x2 + j) = las;
            *(hi + j) = pc;
        }
    }
    // end: read IBs

    printf("\n IBs      longitude    latitude\n");
    for (j = 0; j < nb; j++)
        printf("\n %2d %s %11.6f %11.6f %d", j + 1, (chh + j * 4),
                                             *(x2 + j), *(y2 + j), *(hi + j));

    xmin = 1.0e20;
    xmax = -1.0e20;
    for (j = 0; j < np1; j++) {
        xmin = (xmin < x[j]) ? xmin : x[j];
        xmax = (xmax > x[j]) ? xmax : x[j];
    }

    xmin = xmin - (xmax - xmin) * 0.3;
    xmax = xmax + (xmax - xmin) * 0.3;
    ymin = 1.0e20;
    ymax = -1.0e20;
    
    for (j = 0; j < np1; j++) {
        ymin = (ymin < y[j]) ? ymin : y[j];
        ymax = (ymax > y[j]) ? ymax : y[j];
    }

    ymin = ymin - (ymax - ymin) * 0.3;
    ymax = ymax + (ymax - ymin) * 0.3;
    
    dd = (xmax - xmin) * 0.01;
    
    plsdev("xwin");
    plspage(100.0, 100.0, 1000, 750, 30, 30);
    
    plinit();
    plenv(xmin, xmax, ymin, ymax, 0, 0);
    
    plcol0(COLOR);
    
    pllab("longitude", "latitude", "PSs o IBs +");
    plpoin(np1, x, y, SYM2);
    plpoin(np2, x2, y2, SYM1);

    nsel = 0;

    do {
        plGetCursor( & gin);

        // delete the last pair
        if (gin.button == 2 && option == 1) {
            if (nsel > 0) {
                nsel--;
                is = sel1[nsel];
                selplot(x, y, is, SYM2, COLOR);
                is = sel2[nsel];
                selplot(x2, y2, is, SYM1, COLOR);
            }
            continue;
        }
        if (gin.button == 1) {
            do {
                while (gin.button != 1)
                    plGetCursor( & gin);
                is = psearch(np1, x, y, dd, nsel, sel1);
                if (is == -1)
                    plGetCursor(&gin);

            } while (is == -1);

            sel1[nsel] = is;
            seld[is] = 1;
            selplot(x, y, is, SYM2, SCOLOR);

            //    coupling    
            if (option == 1) {
                gin.button = 0;
                //      printf("1: choice point from the 2nd set (stn,+)\n ");
                do {
                    while (gin.button != 1)
                        plGetCursor( & gin);
                    is = psearch(np2, x2, y2, dd, nsel, sel2);
                    if (is == -1)
                        plGetCursor( & gin);

                } while (is == -1);

                sel2[nsel] = is;
                nsel++;
                selplot(x2, y2, is, SYM1, SCOLOR);
            }
        }
    } while (gin.button != 3);

    //------------------------------------------

    if (option == 0 && find_str(argv[2], ".stn") == 1) {
        rewind(inp);

        nd = np;
        printf("\n\n                 modified \n PSs      longitude    latitude\n\n");

        for (j = 0; j < np; j++) {
            fgets(buf, l, inp);
            if (seld[j] == 0) {
                printf(" %2d     %11.6f %11.6f\n", j + 1, x[j], y[j]);
                fprintf(tem, "%s", buf);
                nd--;
            }
        }
        strcpy(buf, "tempfile.xys");
        remove(argv[1]);
        rename(buf, argv[1]);

        printf("\n deleted   PSs: %2d\n", nd);
        fprintf(log, "\n deleted   PSs: %2d", nd);

    } else remove("tempfile.xys");
    // end if else

    if (option == 1 && find_str(argv[2], ".stn") == 1) {
        for (j = 0; j < nsel; j++) {
            i1 = sel1[j];
            i2 = sel2[j];
            x2[i2] = x[i1];
            y2[i2] = y[i1];
        }

        printf("\n\n                identified\n IBs      longitude    latitude\n");
        for (j = 0; j < np2; j++) {
            printf("\n %2d %s %11.6f %11.6f %d", j + 1, (chh + j * 4),
                                                 x2[j], y2[j], hi[j]);
            fprintf(tem, "%s %11.6f %11.6f %d\n", (chh + j * 4),
                                                  x2[j], y2[j], hi[j]);
        }
    }

    plend();

    printf("\n\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +             end  ib_graphic_identify               +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    exit(0);

} // end graphic_identify

/******************************************************************************
* IDENTIFY_PWR                                                               *
*                                                                            *
* The (super) master date have to by given in the directory name as          *
* "INSAR_yyyymmdd". Reading the pwr data from *.mli files, those pwr values  *
* are summed which values // are larger than the given trseshold and can     *
* be found in the scaled-up rectangle // around the IBs network. Those PSs   *
* are IBs candidates which pwr values are larger // than the trheshold and   *
* can be found in all *.mli images. The PSs with largest // average pwr      *
* values inside the search radius around IBs are identified.                 *
*****************************************************************************/

int identify_pwr(int argc, char * argv[]) {

    char cwd[1024]; // name of working directory

    DIR * data; // id of data directory name
    struct dirent * inp; // directory pointer: inp->d_reclen, inp->d_name

    FILE *lat, *lon, *hgt, // id of Gamma geometric input files
         *pwr,             // id of Gamma *.mli files contaning scaled power values
         *par,             // id of Gamma master.mli.par file
         *stn,             // id of *.stn network file
         *sto,             // id of identified station file (*.sta, *.std or *.ngv)
         *log;

    long int ii, jj,   // long indices
             lin, col, // lines and coloumns of input files 
             date;     // reference date of files 

    int i, j, n, b, e; // indices

    fname *infs; // names of mli files
    pixpwr *sub, // structure of selected data in one *.mli file
           *all; // structure of selected data in all *.mli file

    cxy *stnf; // structure of identified station file

    int no, // number of mli files
        ns, // number of sub records
        ni, // number of actual record
        na, // number of all records
        nib; // number of IBs

    char *files; // buffer for different file names
    char *ddir, // directory name
         *c; // "." if the actual and full path input directory
             // is the same

    float suf, // scale-up factor of rectangle around IBs network
          pwrt, // pwr treshold (0.2-0.3)
          pwrs, // sum of pwr
          srad; // search radius around IBc (40 m)

    float lamn, lamx, // min and max longitudes and latitudes
          fimn, fimx; // of rectangle

    // input data variables
    float latc, lonc, hgtc, pwrd;

    float fi, la,       // latitude and longitude
          fim, lam,     // central latitude and longitude of rectangle  
          he,           // height of IB
          dist, distn;  // distaces beetwin PSs and IBs

    char mdate[9], // master date
         shid[4],  // IB short identifier   -  "IB1"   
         name[21], // name of IB            -  "Deak_str"
         lats[20], // latitude  of IB       -  "47-12-43.12345"
         lons[20]; // longitude of IB       -  "16-14-21.12345" 
    
    int id; // rule of IB (1 - reference, 0 - moving) 

    printf("\n\n");

    printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\
          \n +                       IDENTIFY_PWR                         +\
          \n +     identify PSs as IBs using average pwr of mli files     +\
          \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    if (argc - Minarg < 6) {
        printf("  usage:                                                       \n\
              \n  isign identify_pwr /full_path/ network.stn 1.5 0.3 40.0      \n\
              \n  /full_path/ - full_path of GAMMA preproc directory           \
              \n                (or . in GAMMA preproc directory)              \
              \n  network.stn - IBs network file copied into working directory \
              \n          1.5 - scale-up factor of network search rectangle    \
              \n          0.3 - minimum pwr treshold                           \
              \n         40.0 - search radius (m) between PSs and IBs          \n\
              \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");
        return (1);
    }

    if ((log = fopen("ib_identify_pwr.log", "wt")) == NULL) {
        printf("\n\n Cannot create LOG file !\n\n");
        exit(1);
    }

    for (i = 0; i < argc; i++)
        fprintf(log, " %s", argv[i]);

    //  printf("\n\n");
    //  for(i=0;i<argc;i++)
    //  printf("%d  %s\n",i,argv[i]);
    //  printf("\n\n");

    // --------------------------------------------------------------------
    //         0             1          2       3      4    5   
    // ./ib_identify_pwr  fullpath station.stn scale  pwrt srad

    // --------------------------------------------------------------------
    if ((files = (char * ) malloc(1024 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate files\n\n");
        exit(1);
    }
    if ((ddir = (char * ) malloc(1024 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate ddir\n\n");
        exit(1);
    }

    // --------------------------------------------------------------------
    c = argv[1];
    if ( * c == '.') {
        getcwd(cwd, sizeof(cwd));
        sprintf(ddir, "%s%s", cwd, "/");
    } else ddir = argv[1];

    if (find_str(ddir, "INSAR_") == 0) {
        printf("\n ERROR: INSAR_yyyymmdd is not included in directory name\n\n");
        exit(1);
    }

    master_date(argv[1], mdate); // master date

    sprintf(files, "%s%s.mli.par", ddir, mdate);

    printf(" master date: %s\n\n", mdate);
    fprintf(log, "\n\n master date: %s\n\n", mdate);

    if ((par = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    while (fgets(cwd, 1024, par) > 0) {
        if (find_str(cwd, "range_samples:") == 1)
            sscanf(cwd, "%s %ld", files, & col);
        if (find_str(cwd, "azimuth_lines:") == 1)
            sscanf(cwd, "%s %ld", files, & lin);
    }

    printf(" lines:  %6ld \n columns: %ld  of Gamma files\n", lin, col);
    fprintf(log, " lines:  %6ld \n columns: %ld  of Gamma files\n", lin, col);

    sscanf(argv[3], "%f", & suf); // scale-up
    printf("\n scale-up: %6.2f\n", suf);
    fprintf(log, "\n scale-up: %6.2f\n", suf);
    sscanf(argv[4], "%f", & pwrt); // pwr treshold
    printf(" pwr trhs: %6.2f\n", pwrt);
    fprintf(log, " pwr trhs: %6.2f\n", pwrt);
    sscanf(argv[5], "%f", & srad); // search radius
    printf(" srch rad: %6.2f\n", srad);
    fprintf(log, " srch rad: %6.2f\n", srad);

    sprintf(files, "%s", argv[2]); // stn file
    if ((stn = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    //-----------------------------------------------------------------------
    // define scaled rectangle of ib network

    n = 0;
    while (fscanf(stn, "%s %s %s %s %f %d", shid, name, lats, lons, & he, & id) > 0)
        n++;
    rewind(stn);

    if ((stnf = (cxy * ) malloc(n * sizeof(cxy))) == NULL) {
        printf("\n\n Not enough memory to allocate stnf\n\n");
        exit(1);
    }
    for (i = 0; i < n; i++)
        if (((stnf + i)->ic = (char * ) malloc(4 * sizeof(char))) == NULL) {
            printf("\n\n Not enough memory to allocate stnf.ic\n\n");
            exit(1);
        }

    lamn = fimn = 360.0;
    lamx = fimx = lam = fim = 0.0;
    nib = 0;

    while (fscanf(stn, "%s %s %s %s %f %d", shid, name, lats, lons, & he, & id) > 0) {
        fi = hms_rad(lats) * C;
        la = hms_rad(lons) * C;

        sprintf(stnf[nib].ic, "%s", shid);
        stnf[nib].la = la;
        stnf[nib].fi = fi;
        stnf[nib].pc = id;

        if (la < lamn) lamn = la;
        if (la > lamx) lamx = la;
        if (fi < fimn) fimn = fi;
        if (fi > fimx) fimx = fi;
        nib++;
    }
    lam = (lamn + lamx) / 2.0;
    fim = (fimn + fimx) / 2.0;
    fi = fimx - fim;
    la = lamx - lam;
    lamn = lam - la * suf;
    lamx = lam + la * suf;
    fimn = fim - fi * suf;
    fimx = fim + fi * suf;

    printf("\n search area:  %9.6f %9.6f\n               %9.6f %9.6f\n",
        lamn, lamx, fimn, fimx);
    fprintf(log, "\n search area:  %9.6f %9.6f\n               %9.6f %9.6f\n",
        lamn, lamx, fimn, fimx);

    // end: define scaled rectangle of ib network

    printf("\n input files:\n\n");
    fprintf(log, "\n input files:\n\n");

    sprintf(files, "%s", argv[2]); // stn file
    printf(" %s\n\n", files);
    fprintf(log, " %s\n\n", files);

    sprintf(files, "%s%s.lat", ddir, mdate); // lat file
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((lat = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s%s.lon", ddir, mdate); // lon file
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((lon = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s%s.hgt", ddir, mdate); // hgt file
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((hgt = fopen(files, "rb")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s%s.mli.par", ddir, mdate);
    printf(" %s\n\n", files); // proc_par file
    fprintf(log, " %s\n\n", files);

    // read mli file names from dir data

    if ((infs = (fname * ) malloc(1 * sizeof(fname))) == NULL) {
        printf("\n\n Not enough memory to allocate infs\n\n");
        exit(1);
    }
    if ((infs->fn = (char * ) malloc(14 * sizeof(char))) == NULL) {
        printf("\n\n Not enough memory to allocate infs->fn\n\n");
        exit(1);
    }

    no = 0;
    data = opendir(ddir);
    while (inp = readdir(data)) {
        //   printf(" %d %s\n",inp->d_reclen,inp->d_name );
        if ((inp->d_reclen == 32) && (find_str(inp->d_name, ".mli") == 1)) {
            strcpy((infs + no)->fn, inp->d_name);
            no++;
            if ((infs = (fname * ) realloc(infs, (no + 1) * sizeof(fname))) == NULL) {
                printf("\n\n Not enough memory to reallocate infs\n\n");
                exit(1);
            }
            if (((infs + no)->fn = (char * ) malloc(14 * sizeof(char))) == NULL) {
                printf("\n\n Not enough memory to allocate infs->fn\n\n");
                exit(1);
            }
        }
    }
    closedir(data);

    //  printf ("\n");
    //  for(i=0;i<no;i++)
    //  printf(" %s\n",(infs+i)->fn);

    // end: read mli file names from dir data

    if ((all = (pixpwr * ) malloc(1 * sizeof(pixpwr))) == NULL) {
        printf("\n\n Not enough memory to allocate infs\n\n");
        exit(1);
    }
    if ((sub = (pixpwr * ) malloc(1 * sizeof(pixpwr))) == NULL) {
        printf("\n\n Not enough memory to allocate infs\n\n");
        exit(1);
    }

    // select all candidates inside the rectangle with given pwr treshold

    na = 0;
    for (i = 0; i < no; i++) {
        sprintf(files, "%s%s", ddir, (infs + i)->fn);

        if ((pwr = fopen(files, "rb")) == NULL) {
            printf("\n %s data file not found ! \n\n", files);
            exit(1);
        }

        printf(" %s\n", files);
        fprintf(log, " %s\n", files);

        sscanf((infs + i)->fn, "%8ld", & date);

        ns = 0;
        for (ii = 0; ii < lin; ii++)
            for (jj = 0; jj < col; jj++) {
                fread( &latc, sizeof(float), 1, lat);
                fswap( &latc);
                fread( &lonc, sizeof(float), 1, lon);
                fswap( &lonc);
                fread( &hgtc, sizeof(float), 1, hgt);
                fswap( &hgtc);
                fread( &pwrd, sizeof(float), 1, pwr);
                fswap( &pwrd);

                if (   latc > fimn && latc < fimx
                     && lonc > lamn && lonc < lamx && pwrd >= pwrt) {
                    sub[ns].azi = ii;
                    sub[ns].rng = jj;
                    sub[ns].lat = latc;
                    sub[ns].lon = lonc;
                    sub[ns].hgt = hgtc;
                    sub[ns].pwr = pwrd;
                    ns++;

                    if ((sub = (pixpwr * ) realloc(sub, (ns + 1) * sizeof(pixpwr))) == NULL) {
                        printf("\n\n Not enough memory to reallocate sub\n\n");
                        exit(1);
                    }

                }
            } // for "lin" "col"

        ni = na;
        na = na + ns;
        if ((all = (pixpwr * ) realloc(all, na * sizeof(pixpwr))) == NULL) {
            printf("\n\n Not enough memory to reallocate sub\n\n");
            exit(1);
        }

        for (j = 0; j < ns; j++) {
            all[j + ni].azi = sub[j].azi;
            all[j + ni].rng = sub[j].rng;
            all[j + ni].lat = sub[j].lat;
            all[j + ni].lon = sub[j].lon;
            all[j + ni].hgt = sub[j].hgt;
            all[j + ni].pwr = sub[j].pwr;
        }

        fclose(pwr);
        rewind(lat);
        rewind(lon);
        rewind(hgt);

    } // for "no"

    fclose(lat);
    fclose(lon);
    fclose(hgt);

    // end: select all candidates inside the rectangle with given pwr treshold
    // ----------------------------------------------------------------------
    // ----------------------------------------
    // sorting by ascendind azi and then by rng

    sort_azi_pwr(all, na);

    b = 1;
    for (i = 0; i < na; i++) {
        if (all[i].azi != all[b - 1].azi) {
            e = i;
            sort_rng_pwr(all, b, e);
            b = e + 1;
        }
    }
    sort_rng_pwr(all, b, na);

    // end: sorting
    // ----------------------------------------
    //----------------------------------------- 
    // select the same pixels in all mli files

    free(sub);
    if ((sub = (pixpwr * ) malloc(1 * sizeof(pixpwr))) == NULL) {
        printf("\n\n Not enough memory to allocate infs\n\n");
        exit(1);
    }

    e = 0;
    b = 1;
    printf("\n");
    fprintf(log, "\n");

    for (i = 1; i < na; i++) {
        if (all[i - 1].azi == all[i].azi && all[i - 1].rng == all[i].rng) b++;
        else {
            if (b == no) {
                sub[e].lon = all[i - 1].lon;
                sub[e].lat = all[i - 1].lat;
                sub[e].hgt = all[i - 1].hgt;
                sub[e].rng = all[i - 1].rng;
                sub[e].azi = all[i - 1].azi;
                e++;
                if ((sub = (pixpwr * ) realloc(sub, (e + 1) * sizeof(pixpwr))) == NULL) {
                    printf("\n\n Not enough memory to reallocate sub\n\n");
                    exit(1);
                }
            }
            b = 1;
        }
    }
    if (b == no) {
        sub[e].lon = all[i - 1].lon;
        sub[e].lat = all[i - 1].lat;
        sub[e].hgt = all[i - 1].hgt;
        sub[e].rng = all[i - 1].rng;
        sub[e].azi = all[i - 1].azi;
        e++;
    }

    // end: select the same pixels in all mli files
    // --------------------------------------------
    // --------------------------------------------
    // estimate  cumulative  pwr of selected pixels

    for (i = 0; i < e; i++) {
        pwrs = 0.0;
        for (j = 0; j < na; j++)
            if (sub[i].azi == all[j].azi && sub[i].rng == all[j].rng) pwrs = pwrs + all[j].pwr;
        sub[i].pwr = pwrs;
        printf("%3d  %9.6f %9.6f %5.1f %5.1f %6ld %6ld\n", i, sub[i].lon, sub[i].lat, sub[i].hgt,
            sub[i].pwr, sub[i].rng, sub[i].azi);
        fprintf(log, "%3d  %9.6f %9.6f %5.1f %5.1f %6ld %6ld\n", i, sub[i].lon, sub[i].lat, sub[i].hgt,
            sub[i].pwr, sub[i].rng, sub[i].azi);
    }
    printf("\n selected pixels: %3d\n\n", e);
    fprintf(log, "\n selected pixels: %3d\n\n", e);

    printf(" output files:\n\n");
    fprintf(log, "\n output files:\n\n");

    printf(" ib_identify_pwr.log\n");
    fprintf(log, " ib_identify_pwr.log\n");

    // end: cumulative  pwr estimations
    // --------------------------------------------
    // --------------------------------------------
    // identified stn file

    sprintf(files, "%s", argv[2]);

    if (find_str(ddir, "asc") == 1 || find_str(ddir, "ASC") == 1) change_f_ext(files, ".sta");
    else
    if (find_str(ddir, "dsc") == 1 || find_str(ddir, "DSC") == 1) change_f_ext(files, ".std");
    else {
        change_f_ext(files, ".ngv");
        printf(" Warning: orientation (_asc_ or _dsc_) is not given in directory\n          change .ngv\n");
    }
    printf(" %s\n\n", files);
    fprintf(log, " %s\n\n", files);

    if ((sto = fopen(files, "wt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    // end: identified stn file
    // --------------------------------------------
    // --------------------------------------------
    // select ib-ps pairs

    printf("          dist  a_pwr    rng    azi\n\n");
    fprintf(log, "          dist  a_pwr    rng    azi\n\n");

    for (i = 0; i < nib; i++) {
        pwrt = distn = 0.0;
        n = -1;
        for (j = 0; j < e; j++) {
            la = (sub[j].lon - stnf[i].la) / C;
            fi = (sub[j].lat - stnf[i].fi) / C;
            dist = acos(cos(la) * cos(fi)) * 6372000.0;
            if (dist <= srad && sub[j].pwr > pwrt) {
                pwrt = sub[j].pwr;
                distn = dist;
                n = j;
            }
        }
        if (n == -1) {
            printf(" %s  is not identified  -  repeat proc with new parameters !\n", stnf[i].ic);
            fprintf(log, " %s  is not identified  -  repeat proc with new parameters !\n", stnf[i].ic);

            fprintf(sto, "%s is not identified\n", stnf[i].ic);
        } else {
            printf(" %s %3d %6.1f %6.1f %6ld %6ld\n", stnf[i].ic, n, distn,
                sub[n].pwr / no, sub[n].rng, sub[n].azi);
            fprintf(log, " %s %3d %6.1f %6.1f %6ld %6ld\n", stnf[i].ic, n, distn,
                sub[n].pwr / no, sub[n].rng, sub[n].azi);
            fprintf(sto, "%s %11.6f %11.6f %d\n", stnf[i].ic, sub[n].lon, sub[n].lat, stnf[i].pc);
        }
    }
    printf("\n");

    // end:  select ib-ps pairs
    // --------------------------------------------

    fclose(stn);
    fclose(sto);
    fclose(par);

    return (1);

} // end: main   

/******************************************************************************
* LOS_SERIES                                                                 *
*                                                                            *
* Prepare los series of Integrated Benchmarks (IBs)                          *
*                                                                            *
* The full path of the StaMPS master directory has to be given.              *
* It is supposed that the name of working directory contains                 *
* _ASC_ , _asc_ , _DSC_ or _dsc_ characters, otherwise "ngv"                 *
* will be used in the name of results file, that have to be changed later.   *
* The master date "yyyymmdd" is also subtracted from the master directory    *
* name. The "date.txt" and "ps_u-dm.i.xy" files, where i vary from 1 to n    *
* number of processed images, are stored in the master directory.            *
* "date.txt" contains the increasing dates from i = 1 to n.                  *
* Taking the identified IBs coordinates the relevant los data in the         *
* ps_u-dm.i.xy files are searched and uploaded to memory.                    *
* In the case of master IB the los data are changed to zero.                 *
* Finally the reference IB los data are subtracted from the others.          *
* If no reference IB is given the last IB is treated as reference.           *
* To begin the data series in the first epoch, the los of first date are     *
* also subtracted from the others.                                           *
* The dates are changed to decimal year taking into account the time of      *
* measurements given in decimal hours.                                       *
*****************************************************************************/

int los_series(int argc, char * argv[]) {

    int i, j,
        nep, // number of epochs
        nib, // nunber of IBs
        nr; // reference IB

    int pc, // processing code
        sd; // scene direction: 1 asc, 2 dsc, 0 ngv (not given)

    char *ext[3] = {
        "ngv",
        "asc",
        "dsc"
    };

    double la, // latitude  deg.
           fi, // longitude deg
           los, // los       mm
           h; // decimal hour of measurements

    double *los_tab;

    epoch *meas;
    double date, master;

    cxy *ibsd; // arguments of IBs

    char *inpf, // full path and derived files
         *outf, // output files
         *logf = "series.log", * buf; // multy purpose buffer

    FILE * inp, * inb, * out, * log;

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                   LOS_SERIES                       +\
            \n + prepare los series of  Integrated Benchmarks (IBs) +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    if (argc - Minarg < 3) {
        printf("\n  usage:                                               \
                \n        los_series /full-path_to/ network.sti h.hh     \n\
                \n        /full-path_to/ - /.../INSAR_YYYYMMDD/  (PS)    \
                \n                       - /.../SMALL_BASELINES/         \
                \n                       - /.../MERGED/                  \
                \n           network.sti - file of identified IBs        \
                \n                h.hh   - dec. hour of SAR observation \n\
                \n           ( other files are automatically opened )    \
                \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

        exit(1);
    }

    i = strlen(argv[1]);

    if ((inpf = (char * ) malloc((i + 20) * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate INPF");
        exit(1);
    }
    if ((outf = (char * ) malloc((i + 20) * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate INPF");
        exit(1);
    }

    if ((buf = (char * ) malloc((i + 20) * sizeof(char))) == NULL) {
        printf("\n Not enough memory to allocate BUF");
        exit(1);
    }

    master = master_date_los(argv[1]); // subtract master epoch
    strcpy(buf, argv[1]);
    sd = slc_direction(buf); // subtract slc direction

    sprintf(inpf, "%s%s", argv[1], "date.txt");
    if ((inp = fopen(inpf, "rt")) == NULL) // open date file
    {
        printf("\n %s file not found ! ", inpf);
        exit(1);
    }

    if ((inb = fopen(argv[2], "rt")) == NULL) // open  IBs file
    {
        printf("\n %s file not found ! ", argv[2]);
        exit(1);
    }

    if ((log = fopen(logf, "wt")) == NULL) // open log file
    {
        printf("\n %s file not found ! ", logf);
        exit(1);
    }

    for (i = 0; i < argc; i++)
        fprintf(log, " %s", argv[i]);

    nib = 0;
    while (fgets(buf, 80, inb) > 0) nib++; // nunber of IBs

    // memory allocation
    // for ibsd

    if ((ibsd = (cxy * ) malloc(nib * sizeof(cxy))) == NULL) {
        printf("\n Not enough memory to allocate IBSD");
        exit(1);
    }
    for (i = 0; i < nib; i++) {
        if (((ibsd + i)->ic = (char * ) malloc(4 * sizeof(char))) == NULL) {
            printf("\n Not enough memory to allocate IBSD.IC");
            exit(1);
        }
    }
    rewind(inb);
    nr = -1;
    for (i = 0; i < nib; i++) // upload IB data to memory
    {
        fscanf(inb, "%s %lf %lf %d", buf, & la, & fi, & pc);
        sprintf((ibsd + i)->ic, "%s", buf);
        (ibsd + i)->fi = fi;
        (ibsd + i)->la = la;
        (ibsd + i)->pc = pc;
        if (pc == 1) nr = i;

        //printf("%d %s %lf %lf %d\n",i,(ibsd+i)->ic,(ibsd+i)->la,(ibsd+i)->fi,(ibsd+i)->pc);

    }
    fclose(inb);

    if ((nr >= 0) && (nr != nib - 1)) // reference IB will be the last
        swap_ibs(nib, nr, ibsd); // othervise the last IB will be the reference

    sscanf(argv[3], "%lf", & h);
    printf("\n observation: %6.3lf hours", h);
    fprintf(log, "\n\n observation time: %6.3lf hours", h);

    printf("\n\n IB input file: %s\n", argv[2]);
    fprintf(log, "\n\n IB input file: %s\n", argv[2]);
    printf("\n");
    fprintf(log, "\n");

    for (i = 0; i < nib; i++) // upload the IBs data to memory
    {
        printf(" %s %lf %lf %d\n", (ibsd + i)->ic, (ibsd + i)->la,
                                   (ibsd + i)->fi, (ibsd + i)->pc);
        fprintf(log, " %s %lf %lf %d\n", (ibsd + i)->ic, (ibsd + i)->la,
                                         (ibsd + i)->fi, (ibsd + i)->pc);
    }

    printf("\n input files:\n\n %s\n", inpf);
    fprintf(log, "\n input files:\n\n %s\n", inpf);

    nep = 0;
    while (fscanf(inp, "%lf", & date) > 0) nep++; // number of epochs
    rewind(inp);

    if ((meas = (epoch * ) malloc(nep * sizeof(epoch))) == NULL) {
        printf("\n Not enough memory to allocate INPF");
        exit(1);
    }
    if ((los_tab = (double * ) malloc(nep * nib * sizeof(double))) == NULL) {
        printf("\n Not enough memory to allocate LOS TAB");
        exit(1);
    }

    for (i = 0; i < nep; i++) // upload epochs to memory
    {
        fscanf(inp, "%lf", & date);
        (meas + i)->e = date;
        (meas + i)->yr = date / 10000.0;
        (meas + i)->mn = (date - (meas + i)->yr * 10000.0) / 100.0;
        (meas + i)->dy = date - (meas + i)->yr * 10000.0 - (meas + i)->mn * 100.0;
    }
    fclose(inp);

    // ------------------------------------------------------------------
    // fill up los table

    // epochs
    for (i = 0; i < nep; i++) {
        sprintf(inpf, "%s%s%d%s", argv[1], "ps_u-dm.", i + 1, ".xy");

        printf(" %s\n", inpf);
        fprintf(log, " %s\n", inpf);

        if ((inp = fopen(inpf, "rt")) == NULL) {
            printf("\n %s file not found ! ", inpf);
            exit(1);
        }
        
        // IBs
        for (j = 0; j < nib; j++)  {
            do {
                if (fscanf(inp, " %lf %lf %lf", & la, & fi, & los) <= 0) {
                    printf("\n\n PS not found !\n\n");
                    exit(1);
                }
            } while ((la - (ibsd + j)->la) != 0.0 && (fi - (ibsd + j)->fi) != 0.0);

            //     if( master==(meas+i)->e ) los=0.0;
            * (los_tab + i * nib + j) = los;
            rewind(inp);
        } // for IBs
        fclose(inp);
    } // for epochs

    for (i = 0; i < nep; i++) {
        printf("\n ");
        for (j = 0; j < nib; j++) printf(" %10.6lf", * (los_tab + i * nib + j));
    }

    // ------------------------------------------------------------------

    decimal_year(nep, meas, h); // convert to decimal:  (meas+i)->yrd
    
    // substract the reference IB
    for (i = 0; i < nep; i++) 
        for (j = 0; j < (nib - 1); j++)
            *(los_tab + i * nib + j) -= *(los_tab + i * nib + nib - 1);

    // substract the first los
    for (i = nep - 1; i >= 0; i--) 
        for (j = 0; j < (nib - 1); j++)
            *(los_tab + i * nib + j) -= * (los_tab + j);

    printf("\n            ");
    fprintf(log, "\n            ");
    
    for (j = 0; j < (nib - 1); j++) {
        printf(" %s-%s", (ibsd + j)->ic, (ibsd + nib - 1)->ic);
        fprintf(log, " %s-%s", (ibsd + j)->ic, (ibsd + nib - 1)->ic);
    }
    
    printf("\n");
    fprintf(log, "\n");

    for (i = 0; i < nep; i++) {
        printf("\n %11.6lf", (meas + i)->yrd);
        fprintf(log, "\n %11.6lf", (meas + i)->yrd);
        for (j = 0; j < nib - 1; j++) {
            printf(" %7.2lf", * (los_tab + i * nib + j));
            fprintf(log, " %7.2lf", * (los_tab + i * nib + j));
        }
    }
    
    printf("\n\n output files:\n\n");
    fprintf(log, "\n\n output files:\n\n");

    for (j = 0; j < (nib - 1); j++) {
        sprintf(outf, "%s-%s_%s.los", (ibsd + j)->ic, (ibsd + nib - 1)->ic,
                                      ext[sd]);

        printf(" %s\n", outf);
        fprintf(log, " %s\n", outf);
        
        // open output file
        if ((out = fopen(outf, "wt")) == NULL) {
            printf("\n %s file not found ! ", outf);
            exit(1);
        }

        for (i = 0; i < nep; i++)
            fprintf(out, "%8.0lf  %11.6lf %7.2lf\n", (meas + i)->e,
                (meas + i)->yrd, * (los_tab + i * nib + j));
        fclose(out);
    }

    if (ext[sd] == "ngv") {
        printf("\n Warning ! change the output file names\
                \nto distinguish between asc and dsc data !\n");
        fprintf(log, "\n Warning ! change the output file names\
                      \n to distinguish between asc and dsc data !\n");
    }

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\
            \n +                 end   los_series                   +\
            \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    return 0;
}
/******************************************************************************
* SELECT_STN
* Reading the ps_data.xy StaMPS file those PSs are selected which are        *
* in the area of scaled up-rectangle including the IBs network.              *
*****************************************************************************/

int select_stn(int argc, char * argv[]) {

    int i, j, n;

    cxy * stnf; // station file

    FILE * stn, // id input station file
        * xy, // id input ps_data file
        * xys, // id outpot *_data.xys file
        * log;

    char files[1024], // buffer for different file names
        cwd[1024], // buffer of working directory
        wdir[1024]; // buffer of full path input directory

    float fimn, fimx, lamn, lamx; // corners of scaled up IB network rectangle

    float fi, la, // latitude & longitude
    fim, lam, // central latitude & longitude of rectangle
    he, dhe, v, // height, height error & velocity of ps_data.xy
    suf; // scale_up of network rectangle

    char * c; // "." if the actual and full path input directory
    // is the same

    char shid[4], // IB short identifier   -  "IB1"   
        name[21], // name of IB            -  "Deak_str"
        lats[20], // latitude  of IB       -  "47-12-43.12345"
        lons[20]; // longitude of IB       -  "16-14-21.12345" 
    int id; // roule of IB (1 - reference, 0 - moveing) 

    // --------------------------------------------
    //         0            1          2        3  
    // ./ps_select_stn  fullpath  station.stn scale
    // --------------------------------------------

    //  for(i=0;i<argc;i++)
    //  printf("%d  %s\n",i,argv[i]);
    //  printf("argc %d",argc);
    //  printf("\n\n");

    printf("\n\n");

    printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf(" +                      SELECT_STN                            +\n");
    printf(" +  select PSs in the scaled-up rectangle around IB network   +\n");
    printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    if (argc - Minarg < 3) {
        printf("   usage:        isign select_stn /full_path/ network.stn 1.5 \n\n");
        printf("   /full_path/ - full_path of StaMPS solution directory        \n");
        printf("                 (or . in StaMPS solution directory)           \n");
        printf("   network.stn - IBs network file copied into working directory\n");
        printf("           1.5 - scale-up factor of search rectangle           \n\n");
        printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");
        return (1);
    }

    c = argv[1];
    if ( * c == '.') {
        getcwd(cwd, sizeof(cwd));
        sprintf(wdir, "%s%s", cwd, "/");
    } else sprintf(wdir, "%s", argv[1]);

    if ((log = fopen("ps_select_stn.log", "wt")) == NULL) {
        printf("\n\n Cannot create LOG file !\n\n");
        exit(1);
    }

    //  printf(" %s\n",wdir);

    // -------------------------------------------------------------
    // open datafiles

    fprintf(log, "\n %s %s %s %s\n\n", argv[0], argv[1], argv[2], argv[3]);

    sscanf(argv[3], "%f", & suf); // scale-up
    printf(" csale-up: %5.2f\n\n", suf);
    fprintf(log, " csale-up: %5.2f\n\n", suf);

    printf(" input files:\n\n");
    fprintf(log, " input files:\n\n");

    sprintf(files, "%s%s", wdir, "ps_data.xy"); // ps_data.xy
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((xy = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    sprintf(files, "%s", argv[2]); // stn file
    printf(" %s\n", files);
    fprintf(log, " %s\n", files);
    if ((stn = fopen(files, "rt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    printf("\n output files:\n\n");
    fprintf(log, "\n output files:\n\n");

    if (find_str(wdir, "asc") == 1 || find_str(wdir, "ASC") == 1)
        sprintf(files, "%s", "asc_stamps.xys");
    else
    if (find_str(wdir, "dsc") == 1 || find_str(wdir, "DSC") == 1)
        sprintf(files, "%s", "dsc_stamps.xys");
    else {
        sprintf(files, "%s", "ngv_stamps.xys");
        printf(" Warning: orientation (asc or dsc) is not given \n");
        printf("          change ngv for asc or dsc\n\n");
        fprintf(log, " Warning: orientation (asc or dsc) is not given \n");
        fprintf(log, "          change ngv for asc or dsc\n\n");
    }

    printf(" %s\n", files); // asc/dsc_data.xys
    fprintf(log, " %s\n", files);
    if ((xys = fopen(files, "wt")) == NULL) {
        printf("\n %s data file not found ! \n\n", files);
        exit(1);
    }

    printf(" ps_select_stn.log\n\n");
    fprintf(log, " ps_select_stn.log\n\n");

    // end: open data files
    // ---------------------------------------------------------------
    //-----------------------------------------------------------------------
    // define scaled rectangle of IB network

    n = 0;
    while (fscanf(stn, "%s %s %s %s %f %d", shid, name, lats, lons, & he, & id) > 0)
        n++;
    rewind(stn);

    if ((stnf = (cxy * ) malloc(n * sizeof(cxy))) == NULL) {
        printf("\n\n Not enough memory to allocate stnf\n\n");
        exit(1);
    }
    for (i = 0; i < n; i++)
        if (((stnf + i)->ic = (char * ) malloc(4 * sizeof(char))) == NULL) {
            printf("\n\n Not enough memory to allocate stnf.ic\n\n");
            exit(1);
        }

    lamn = fimn = 360.0;
    lamx = fimx = lam = fim = 0.0;

    while (fscanf(stn, "%s %s %s %s %f %d", shid, name, lats, lons, & he, & id) > 0) {
        fi = hms_rad(lats) * C;
        la = hms_rad(lons) * C;

        if (la < lamn) lamn = la;
        if (la > lamx) lamx = la;
        if (fi < fimn) fimn = fi;
        if (fi > fimx) fimx = fi;
    }

    lam = (lamn + lamx) / 2.0;
    fim = (fimn + fimx) / 2.0;
    fi = fimx - fim;
    la = lamx - lam;
    lamn = lam - la * suf;
    lamx = lam + la * suf;
    fimn = fim - fi * suf;
    fimx = fim + fi * suf;

    printf(" search area:  %9.6f %9.6f\n               %9.6f %9.6f\n\n",
        lamn, lamx, fimn, fimx);

    fprintf(log, " search area:  %9.6f %9.6f\n               %9.6f %9.6f\n\n",
        lamn, lamx, fimn, fimx);

    // end: define scaled rectangle of ib network
    //-----------------------------------------------------------------------
    //-----------------------------------------------------------------------
    // select candidate

    n = 0;
    while (fscanf(xy, "%f %f %f %f %f", & la, & fi, & v, & he, & dhe) > 0) {
        if (fi > fimn && fi < fimx && la > lamn && la < lamx) {
            fprintf(xys, "%11.6f %11.6f %8.4f %8.4f %8.4f\n", la, fi, v, he, dhe);
            n++;
        }
    }

    printf(" Selected PSs:  %d\n\n", n);
    fprintf(log, " Selected PSs:  %d\n\n", n);

    fclose(stn);
    fclose(xy);
    fclose(xys);

    printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    printf(" +                end    ps_select_stn                        +\n");
    printf(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    return (1);

} // end select_stn

/*****************
 * Main function *
 *****************/

int main(int argc, char **argv)
{
    if (argc < 2) {
        errorln("At least one argument (module name) is required.\
                 \nModules to choose from: %s.", Modules);
        return(-1);
    }
    
    if (Module_Select("select_stn"))
        return select_stn(argc, argv);
    
    else if(Module_Select("graphic_identify"))
        return graphic_identify(argc, argv);
    
    else if(Module_Select("graphic_control"))
        return graphic_control(argc, argv);
    
    else if(Module_Select("los_series"))
        return los_series(argc, argv);

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return(-1);
    }
}
