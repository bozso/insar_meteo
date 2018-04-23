#include <stdio.h>
#include <tgmath.h>
#include <stdlib.h>
#include <string.h>
#include <aux_module.h>

//-----------------------------------------------------------------------------
// AUXILLIARY FUNCTIONS
//-----------------------------------------------------------------------------

static inline int plc(const int i, const int j)
{
    // position of i-th, j-th element  0
    // in the lower triangle           1 2
    // stored in vector                3 4 5

    int k, l;
    k = (i < j) ? i + 1 : j + 1;
    l = (i > j) ? i + 1 : j + 1;
    return((l - 1) * l / 2 + k - 1);
}
// end plc

static void ATA_ATL(const uint m, const uint u, const double *A, const double *L,
                    double *ATA, double *ATL)
{
    // m - number of measurements
    // u - number of unknowns
    uint kk;
    double *buf = Aux_Malloc(double, m);
    
    // row of ATPA
    FOR(ii, 0, u)  {
        FOR(kk, 0, m) {
            buf[kk] = 0.0;

            FOR(ll, 0, m)
                buf[kk] += A[ ll* u + ii ];
        }

        FOR(ll, 0, m)
            ATL[ii] += buf[ll] * L[ll];
        
        // column of ATPA
        FOR(jj, 0, u)  {
            kk = plc(ii, jj);

            FOR(ll, 0, m)
                ATA[kk] += buf[ll] * A[ ll * u + jj ];
        } //end - column of ATPA
    } // end - row of ATPA

    Aux_Free(buf);
}
// end ATA_ATL

static int chole(double *q, const uint n)
{
    // Cholesky decomposition of
    // symmetric positive delatnit
    // normal matrix

    uint ia, l1, ll;
    double a, sqr, sum;

    FOR(ii, 0, n) {
        a =  q[plc(ii, ii)];
        if (a <= 0.0) return(1);
        sqr = sqrt(a);

        FOR(kk, ii, n)
            q[plc(ii, kk)] /= sqr;

        ll = ii + 1;
        if (ll == n) goto end1;

        FOR(jj, ll, n) FOR(kk, jj, n)
            q[plc(jj, kk)] -= q[plc(ii, jj)] * q[plc(ii, kk)];
    }

end1:
    FOR(ii, 0, n) {
        ia = plc(ii, ii); q[ia] = 1.0 / q[ia];
        l1 = ii + 1; if (l1 == n) goto end2;

        FOR(jj, l1, n) {
            sum = 0.0;
            FOR(kk, ii, jj)
                sum += q[plc(ii, kk)] * q[plc(kk, jj)];

            q[plc(ii, jj)] = -sum / q[plc(jj, jj)];
        }
    }

end2:
    FOR(ii, 0, n) FOR(kk, ii, n) {
        sum = 0.0;

        FOR(jj, kk, n)
            sum += q[plc(ii, jj)] * q[plc(kk, jj)];

        q[plc(ii, kk)] = sum;
    }

    return(0);
}
// end chole

static int poly_fit(const uint m, const uint u, const torb *orb,
                    double *X, const char c)
{
    // o(t) = a0 + a1*t + a2*t^2 + a3*t^3  + ...

    double t, t0, L, *A, *ATA, *ATL, mu0 = 0.0;

    t0 = orb[0].t;

    A   = Aux_Malloc(double, u);
    ATA = Aux_Malloc(double, u * (u + 1) / 2);
    ATL = Aux_Malloc(double, u);

    if (c != 'x' OR c != 'y' OR c != 'z') {
        errorln("Invalid option for char \"c\" %c\t Valid options "
               "are: \"x\", \"y\", \"z\"\n", c);
        return(Err_Num);
    }

    FOR(ii, 0, m) {
             if (c == 'x') L = orb[ii].x;
        else if (c == 'y') L = orb[ii].y;
        else if (c == 'z') L = orb[ii].z;

        t = orb[ii].t - t0;

        FOR(jj, 0, u)
            A[jj] = pow(t, 1.0 * jj);

        ATA_ATL(1, u, A, &L, ATA, ATL);
    }

    if (!chole(ATA, u)) {
        fprintf(stderr, "poly_fit: Error - singular normal matrix!");
        exit(Err_Num);
    }
    
    // update of unknowns
    FOR(ii, 0, u) {
        X[ii] = 0.0;

        FOR(jj, 0, u)
            X[ii] += ATA[plc(ii, jj)] * ATL[jj];
    }

    FOR(ii, 0, m) {
             if (c == 'x') L = -orb[ii].x;
        else if (c == 'y') L = -orb[ii].y;
        else if (c == 'z') L = -orb[ii].z;

        t = orb[ii].t - t0;

        FOR(jj, 0, u)
            L += X[jj] * pow(t, (double) jj);

        mu0 += L * L;
    }

    mu0 = sqrt(mu0 / ((double) (m - u)) );

    printf("\n\n          mu0 = %8.4lf", mu0);
    printf("degree of freedom = %d", m - u);
    printf("\n\n         coefficients                  std\n");

    FOR(jj, 0, u)
        printf("\n%2d %23.15e   %23.15e", jj, X[jj],
                                mu0 * sqrt( ATA[plc(jj, jj)] ));
    
    Aux_Free(A); Aux_Free(ATA); Aux_Free(ATL);
    
    return(0);
}
// end poly_fit

static void estim_dominant(const psxys *buffer, const uint ps1, const uint ps2,
                           FILE *ou)
{
    double dist, dx, dy, dz, sumw, sumwve;
    station ps, psd;

    // coordinates of dominant point - weighted mean
    psd.x = psd.y = psd.z = 0.0;

    FOR(ii, 0, ps1 + ps2) {
        ps.lat = buffer[ii].lat * DEG2RAD;
        ps.lon = buffer[ii].lon * DEG2RAD;
        ps.h = buffer[ii].he;

        // compute ps.x ps.y ps.z
        ell_cart( &ps );

        if(ii<ps1) {
            psd.x += ps.x/ps1;
            psd.y += ps.y/ps1;
            psd.z += ps.z/ps1;
        }
        else {
            psd.x += ps.x/ps2;
            psd.y += ps.y/ps2;
            psd.z += ps.z/ps2;
        }
        // sum (1/ps1 + 1/ps2) = 2
    } //end for

   psd.x /= 2.0;
   psd.y /= 2.0;
   psd.z /= 2.0;

   cart_ell( &psd );

   fprintf(ou, "%16.7le %15.7le %9.3lf",
               psd.lon * RAD2DEG, psd.lat * RAD2DEG, psd.h);

    // interpolation of ascending velocities
    sumwve = sumw = 0.0;

    FOR(ii, 0, ps1) {
        ps.lat = buffer[ii].lat * DEG2RAD;
        ps.lon = buffer[ii].lon * DEG2RAD;
        ps.h = buffer[ii].he;

        ell_cart( &ps );

        dx = psd.x - ps.x;
        dy = psd.y - ps.y;
        dz = psd.z - ps.z;
        dist = distance(dx, dy, dz);

        sumw   += 1.0 / dist / dist; // weight
        sumwve += buffer[ii].ve / dist / dist;
    }

    fprintf(ou," %8.3lf", sumwve / sumw);

    // interpolation of descending velocities

    sumwve = sumw = 0.0;

    FOR(ii, ps1, ps1 + ps2) {
        ps.lat = buffer[ii].lat * DEG2RAD;
        ps.lon = buffer[ii].lon * DEG2RAD;
        ps.h = buffer[ii].he;

        ell_cart( &ps );

        dx = psd.x - ps.x;
        dy = psd.y - ps.y;
        dz = psd.z - ps.z;

        dist = distance(dx, dy, dz);


        sumw   += 1.0 / dist / dist; // weight
        sumwve += buffer[ii].ve / dist / dist;
    }
    fprintf(ou," %8.3lf\n", sumwve / sumw);
}
//end estim_dominant

static int cluster(psxys *indata1, const uint n1, psxys *indata2, const uint n2,
                   psxys *buffer, uint *nb, const float dam)
{
    uint kk = 0, jj = 0;
    double dlon, dlat, dd, lon, lat;
    double dm = dam / R_earth * RAD2DEG * dam / R_earth * RAD2DEG;

    // skip selected PSs
    while( (indata1[kk].ni == 0) && (kk < n1) )  kk++;

    lon = indata1[kk].lon;
    lat = indata1[kk].lat;

    // 1 for
    FOR(ii, kk, n1) {

        dlon = indata1[ii].lon - lon;
        dlat = indata1[ii].lat - lat;
        dd = dlon * dlon + dlat*dlat;

        if( (indata1[ii].ni > 0) && (dd < dm) ) {
            buffer[jj] = indata1[ii];
            indata1[ii].ni = 0;
            
            jj++;
            if( jj == *nb) {
                (*nb)++;
                Aux_Realloc(buffer, psxys, *nb);
            }
        } // end if
    } // end  1 for
    // 2 for

    FOR(ii, 0, n2) {
        dlon = indata2[ii].lon - lon;
        dlat = indata2[ii].lat - lat;
        dd = dlon * dlon + dlat * dlat;

        if( (indata2[ii].ni > 0) AND (dd < dm) ) {
            buffer[jj] = indata2[ii];
            indata2[ii].ni = 0;
            
            jj++;
            if(jj == *nb) {
                (*nb)++;
                Aux_Realloc(buffer, psxys, *nb);
            }
        } // end if
    } // end for

    return(jj);
}
// end cluster

static int selectp(const float dam, FILE *in1, const psxy *in2, const uint ni,
                   FILE *ou1)
{
    // The PS"lon1,lat1" is selected if the latrst PS"lon2,lat1"
    // is closer than the sepration distance "dam"
    // The "dam", "lon1,lat1" and "lon2,lat1" are interpreted
    // on spherical Earth with radius 6372000 m

    uint n = 0, // # of selected PSs
         m = 0, // # of data in in1
         ef;
    float lat1, lon1, v1, he1, dhe1, lat2, lon2, da;

    // for faster run
    float dm = dam / R_earth * RAD2DEG * dam / R_earth * RAD2DEG;

    while( fscanf(in1,"%e %e %e %e %e", &lon1, &lat1, &v1, &he1, &dhe1) > 0) {
        ef = 0;
        do {
            lon2 = in2[ef].lon;
            lat2 = in2[ef].lat;

//        da= R * acos( sin(lat1/C)*sin(lat2/C)+cos(lat1/C)*cos(lat2/C)*cos((lon1-lon2)/C) );
//        da= da-dam;

            da =   (lat1 - lat2) * (lat1 - lat2)
                 + (lon1 - lon2) * (lon1 - lon2); // faster run
            da = da - dm;

            ef++;
        } while( (da > 0.0) && ((ef - 1) < ni) );

        if( (ef - 1) < ni )
        {
            fprintf(ou1, "%16.7e %16.7e %16.7e %16.7e %16.7e\n",
                          lon1, lat1, v1, he1, dhe1);
            n++;
        }
        m++;  if( !(m % 10000) ) printf("\n %6d ...", m);
    }

    return(n);
}
// end selectp

static void movements(const double azi1, const double inc1, const float v1,
                      const double azi2, const double inc2, const float v2,
                      float *up, float *east)
{
    double  a1, a2, a3;       // unit vector of sat1
    double  d1, d2, d3;       // unit vector of sat2
    double  n1, n2, n3,  ln;  // 3D vector and its legths
    double  s1, s2, s3,  ls;  // 3D vector and its legths
    double zap, zdp;     // angles in observation plain
    double az, ti, hl;
    double al1,al2,in1,in2;
    double sm, vm;            // movements in observation plain

    al1 = azi1 * DEG2RAD;
    in1 = inc1 * DEG2RAD;
    al2 = azi2 * DEG2RAD;
    in2 = inc2 * DEG2RAD;

//-------------------------

    a1 = -sin(al1) * sin(in1);    // E
    a2 = -cos(al1) * sin(in1);    // N
    a3 =  cos(in1);               // U

    d1 = -sin(al2) * sin(in2);
    d2 = -cos(al2) * sin(in2);
    d3 =  cos(in2);

//-------------------------------------------------------

    // normal vector
    axd(a1, a2, a3, d1, d2, d3, &n1, &n2, &n3);
    ln = sqrt(n1 * n1 + n2 * n2 + n3 * n3);

//-------------------------------------------------------

    n1 /= ln; n2 /= ln; n3 /= ln;

    az = atan(n1 / n2);

    hl = sqrt(n1 * n1 + n2 * n2);
    ti = atan(n3 / hl);

    s1 = -n3 * sin(az);
    s2 = -n3 * cos(az);
    s3 = hl;

    //  vector in the plain
    n1 = s1;
    n2 = s2;
    n3 = s3;

//---------------------------------------

    axd(a1, a2, a3, n1, n2, n3, &s1, &s2, &s3);
    ls = sqrt(s1 * s1 + s2 * s2 + s3 * s3);

    // alfa
    zap=asin(ls);

    axd(d1, d2, d3, n1, n2, n3, &s1, &s2, &s3);
    ls = sqrt(s1 * s1 + s2 * s2 + s3 * s3);

    // beta
    zdp=asin(ls);

    // strike movement
    sm = (v2 / cos(zdp) - v1 / cos(zap)) / (tan(zap) + tan(zdp));

    // tilt movement
    vm = v1 / cos(zap) + tan(zap) * sm;

    // biased Up   component
    *up   = vm / cos(ti);
    // biased East component
    *east = sm / cos(az);

} // end  movement

//-----------------------------------------------------------------------------
// MAIN MODULES
//-----------------------------------------------------------------------------

Mk_Doc(
    data_select,
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "\n +                  ps_data_select                    +"
    "\n + adjacent ascending and descending PSs are selected +"
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    "\n   usage:  daisy data_select in_asc in_dsc out_asc out_dsc separation \n"
    "\n           asc_data.xy  - (1st) ascending  data file"
    "\n           dsc_data.xy  - (2nd) descending data file"
    "\n           100          - (3rd) PSs separation (m)\n"
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

static int data_select(int argc, char **argv)
{
    Aux_CheckArg(data_select, 7);
        
    const char *in_asc  = argv[2],
               *in_dsc  = argv[3],
               *out_asc = argv[4],
               *out_dsc = argv[5];
    
    float max_diff = (float) atof(argv[6]);
    
    println("%s %s %s %s %f", in_asc, in_dsc, out_asc, out_dsc, max_diff);
    
    uint n, ni1, ni2;
    psxy *indata;
    
    FILE *inf_asc, *inf_dsc, *outf_asc, *outf_dsc;
    
    float lon, lat, v, he, dhe;

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                  ps_data_select                    +"
           "\n + adjacent ascending and descending PSs are selected +"
           "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    inf_asc = sfopen(in_asc, "rt");
    inf_dsc = sfopen(in_dsc, "rt");

    outf_asc = sfopen(out_asc, "w+t");
    outf_dsc = sfopen(out_dsc, "w+t");

    println("\n Appr. PSs separation %5.1f (m)", max_diff);

    ni1 = 0;
    while(fscanf(inf_asc, "%e %e %e %e %e", &lon, &lat, &v, &he, &dhe) > 0) ni1++;
    rewind(inf_asc);

    ni2 = 0;
    while(fscanf(inf_dsc, "%e %e %e %e %e", &lon, &lat, &v, &he, &dhe) > 0) ni2++;
    rewind(inf_dsc);

    //  Copy data to memory
    indata = Aux_Malloc(psxy, ni2);

    FOR(ii, 0, ni2)
        fscanf(inf_dsc, "%e %e %e %e %e", &(indata[ii].lon), &(indata[ii].lat),
                                          &v, &he, &dhe);

    println("\n\n %s  PSs %u", in_asc, ni1);

    printf("\n Select PSs ...\n");
    n = selectp(max_diff, inf_asc, indata, ni2, outf_asc);

    rewind(outf_asc);
    rewind(inf_asc);
    rewind(inf_dsc);

    println("\n\n %s PSs %u", out_dsc, n);
    println("\n\n %s PSs %u", in_dsc, ni2);

    Aux_Free(indata);

    // Copy data to memory
    indata = Aux_Malloc(psxy, n);
    
    FOR(ii, 0, n)
        fscanf(outf_asc, "%e %e %e %e %e", &(indata[ii].lon), &(indata[ii].lat),
                                           &v, &he, &dhe);

    printf("\n Select PSs ...\n");
    n = selectp(max_diff, inf_dsc, indata, n, outf_dsc);

    printf("\n\n %s PSs %d\n" , out_dsc, n);

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                end   ps_data_select                +"
           "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    Aux_Free(indata);

    return(0);

}  // end data_select

Mk_Doc(
    dominant,
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "\n +                      ps_dominant                      +"
    "\n + clusters of ascending and descending PSs are selected +"
    "\n +     and the dominant points (DSs) are estimated       +"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

    "\n    usage:  daisy dominant asc_data.xys dsc_data.xys 100\n"
    "\n            asc_data.xys   - (1st) ascending  data file"
    "\n            dsc_data.xys   - (2nd) descending data file"
    "\n            100            - (3rd) cluster separation (m)\n"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

static int dominant(int argc, char **argv)
{
    const char *in_asc, *in_dsc, *out;
    float max_diff;
    
    Aux_CheckArg(dominant, 4);
    
    in_asc = argv[2];
    in_dsc = argv[3];
    out = argv[4];
    
    max_diff = (float) atof(argv[5]);
    
    uint n1, n2,  // number of data in input files
         nb = 2,  // starting number of data in cluster buffer, continiusly updated
         nc,      // number of preselected clusters
         nsc,     // number of selected clusters
         nhc,     // number of hermit clusters
         nps,     // number of selected PSs in actual cluster
         ps1,     // number of PSs from 1 input file
         ps2;     // number of PSs from 2 input file

    psxys *indata1, *indata2, *buffer = NULL;  // names of allocated memories

    FILE *in1, *in2, *ou;

    float lon, lat, he, dhe, ve;

    printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                      ps_dominant                      +"
           "\n + clusters of ascending and descending PSs are selected +"
           "\n +     and the dominant points (DSs) are estimated       +"
           "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    in1 = sfopen(in_asc, "rt");
    in2 = sfopen(in_dsc, "rt");
    ou  = sfopen(out, "w+t");

    println("\n  input: %s\n         %s", in_asc, in_dsc);
    printf("\n output: %s\n\n", out);

    println("\n Appr. cluster size %5.1f (m)", max_diff);
    printf("\n Copy data to memory ...\n");

    n1 = 0;
    while(fscanf(in1, "%e %e %e %e %e", &lon, &lat, &ve, &he, &dhe) > 0) n1++;
    rewind(in1);

    indata1 = Aux_Malloc(psxys, n1);

    FOR(ii, 0, n1) {
         fscanf(in1,"%e %e %e %e %e", &(indata1[ii].lon),
                                      &(indata1[ii].lat),
                                      &(indata1[ii].ve), &he, &dhe);
         indata1[ii].ni = 1;
         indata1[ii].he = he + dhe;
    }
    fclose(in1);

    n2 = 0;
    while(fscanf(in2, "%e %e %e %e %e", &lon, &lat, &ve, &he, &dhe) > 0) n2++;
    rewind(in2);
    
    printf("%d\n", n2);
    
    indata2 = Aux_Malloc(psxys, n2);

    FOR(ii, 0, n2) {
        fscanf(in2, "%e %e %e %e %e", &(indata2[ii].lon),
                                      &(indata2[ii].lat),
                                      &(indata2[ii].ve), &he, &dhe);
        indata2[ii].ni = 2;
        indata2[ii].he = he + dhe;
    }
    fclose(in2);

    printf("\n selected clusters:\n");

    nps = nc = nhc = nsc = 0;

    do {
        nps = cluster(indata1, n1, indata2, n2, buffer, &nb, max_diff);

        ps1 = ps2 = 0;
        FOR(ii, 0, nps) if( buffer[ii].ni == 1) ps1++;
        else            if( buffer[ii].ni == 2) ps2++;

        if( (ps1 * ps2) > 0) {
            estim_dominant(buffer, ps1, ps2, ou);
            nsc++;
        }
        else if ( (ps1 + ps2) > 0 ) nhc++;

        nc++;

        if((nc % 2000) == 0) printf("\n %6d ...", nc);

    } while(nps > 0 );

    printf("\n %6d", nc - 1);

    printf("\n\n hermit   clusters: %6d\n accepted clusters: %6d\n", nhc, nsc);
    printf("\n Records of %s file:\n", out);

    printf("\n longitude latitude  height asc_v dsc_v");
    printf("\n (     degree          m      mm/year )\n");

    printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                   end    ps_dominant                  +"
           "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    Aux_Free(indata1); Aux_Free(indata2);

    return(0);
} // end dominant

Mk_Doc(
    poly_orbit,
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "\n +                     ps_poly_orbit                     +"
    "\n +    tabular orbit data are converted to polynomials    +"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    "\n          usage: daisy poly_orbit asc_master.res 4"
    "\n                 or"
    "\n                    ps_poly_orbit dsc_master.res 4"
    "\n\n          asc_master.res or dsc_master.res - input files"
    "\n          4                                - degree     \n"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

static int poly_orbit(int argc, char **argv)
{
    const char *in_data;
    uint deg_poly;
    
    Aux_CheckArg(poly_orbit, 2);
    
    in_data = argv[2];
    deg_poly = (uint) atoi(argv[3]);
    
    // number of orbit records
    uint ndp;

    // tabular orbit data
    torb *orb;
    double *X;

    char *out = NULL, buf[80], *head = "NUMBER_OF_DATAPOINTS:";

    FILE *in, *ou;

    printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                     ps_poly_orbit                     +"
           "\n +    tabular orbit data are converted to polynomials    +"
           "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    sprintf(out, "%s", in_data);
    change_ext(out, "porb");

    in = sfopen(in_data, "rt");
    ou = sfopen(out, "w+t");

    printf("\n  input: %s",in_data);
    printf("\n output: %s",out);
    println("\n degree: %d",deg_poly);

    while( fscanf(in, "%s", buf) > 0 && strncmp(buf, head, 21) != 0 );
    fscanf(in, "%d", &ndp);

    orb = Aux_Malloc(torb, ndp);
    
    X = Aux_Malloc(double, deg_poly + 1);

    FOR(ii, 0, ndp)
        fscanf(in, "%lf %lf %lf %lf", &(orb[ii].t), &(orb[ii].x),
                                      &(orb[ii].y), &(orb[ii].z));
    fprintf(ou, "%3d\n", deg_poly);
    fprintf(ou, "%13.5f\n", orb[0].t);
    fprintf(ou, "%13.5f\n", orb[ndp - 1].t);

    printf("\n fit of X coordinates:");

    poly_fit(ndp, deg_poly + 1, orb, X, 'x');
    FOR(ii, 0, deg_poly + 1)  fprintf(ou," %23.15e", X[ii]);

    fprintf(ou,"\n");

    printf("\n fit of Y coordinates:");

    poly_fit(ndp, deg_poly + 1, orb, X, 'y');
    FOR(ii, 0, deg_poly + 1)  fprintf(ou," %23.15e", X[ii]);

    fprintf(ou,"\n");

    printf("\n fit of Z coordinates:");

    poly_fit(ndp, deg_poly + 1, orb, X, 'z');
    FOR(ii, 0, deg_poly + 1)  fprintf(ou," %23.15e", X[ii]);

    fprintf(ou,"\n");

    fclose(in);
    fclose(ou);

    printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
             "\n +             end          ps_poly_orbit                +"
             "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    Aux_Free(orb); Aux_Free(X);

    return(0);
}  // end daisy_poly_orbit

Mk_Doc(
    integrate,
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "\n +                       ds_integrate                          +"
    "\n +        compute the east-west and up-down velocities         +"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    "\n usage:                                                      \n"
    "\n    daisy integrate dominant.xyd asc_master.porb dsc_master.porb\n"
    "\n              dominant.xyd  - (1st) dominant DSs data file   "
    "\n           asc_master.porb  - (2nd) ASC polynomial orbit file"
    "\n           dsc_master.porb  - (3rd) DSC polynomial orbit file\n"
    "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

static int integrate(int argc, char **argv)
{
    const char *in_dom, *asc_orb, *dsc_orb, *out;

    uint nn = 0;
    station ps, sat;

    double azi1, inc1, azi2, inc2;
    double ft1, lt1,               // first and last time of orbit files
           ft2, lt2,               // first and last time of orbit files
           *pol1, *pol2;           // orbit polinomials

    uint dop1, dop2;              // degree of orbit polinomials

    float lon, lat, he, v1, v2, up, east;

    FILE *ind, *ino1, *ino2, *ou;
    
    Aux_CheckArg(integrate, 4);
    
    in_dom  = argv[2];
    asc_orb = argv[3];
    dsc_orb = argv[4];
    out     = argv[5];
    
    printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                       ds_integrate                          +"
           "\n +        compute the east-west and up-down velocities         +"
           "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    printf("\n  inputs:   %s\n          %s\n          %s",
           in_dom, asc_orb, dsc_orb);
    println("\n\n outputs:  %s", out);

//-----------------------------------------------------------------------------
    
    ino1 = sfopen(asc_orb, "rt");

    fscanf(ino1, "%d %lf %lf", &dop1, &ft1, &lt1); // read orbit
    dop1++;

    pol1 = Aux_Malloc(double, dop1 * 3);

    ino1 = sfopen(asc_orb, "rt");

    FOR(ii, 0, 3)
        FOR(jj, 0, dop1)
            fscanf(ino1," %lf", &(pol1[ii * dop1 + jj]));

    fclose(ino1);

//-----------------------------------------------------------------------------

    ino2 = sfopen(dsc_orb, "rt");

    fscanf(ino2, "%d %lf %lf", &dop2, &ft2, &lt2); // read orbit
    dop2++;
    
    pol2 = Aux_Malloc(double, dop2 * 3);

    FOR(ii, 0, 3)
        FOR(jj, 0, dop2)
            fscanf(ino2," %lf", &(pol2[ii * dop2 + jj]));

    fclose(ino2);

//-----------------------------------------------------------------------------

    ind = sfopen(in_dom, "rt");
    ou = sfopen(out, "w+t");

    while( fscanf(ind, "%f %f %f %f %f", &lon, &lat, &he, &v1, &v2) > 0 ) {
        ps.lat = lat / 180.0 * M_PI;
        ps.lon = lon / 180.0 * M_PI;
        ps.h = he;

        ell_cart(&ps);

        closest_appr(pol1, dop1, ft1, lt1, &ps, &sat, 1000);
        azim_elev(ps, sat, &azi1, &inc1);

        closest_appr(pol2, dop2, ft2, lt2, &ps, &sat, 1000);
        azim_elev(ps, sat, &azi2, &inc2);

        movements(azi1, inc1, v1, azi2, inc2, v2, &up, &east);

        fprintf(ou,"%16.7e %15.7e %9.3f %7.3f %7.3f\n", lon, lat, he,
                                                        east, up);
        nn++;  if( !(nn % 1000) ) printf("\n %6d ...", nn);
    }

    fclose(ind); fclose(ou);

    printf("\n %6d", nn);
    println("\n\n Records of %s file:", out);
    printf("\n longitude latitude  height  ew_v   up_v");
    printf("\n (     degree          m       mm/year )");


    printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
             "\n +                     end    ds_integrate                     +"
             "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    Aux_Free(pol1); Aux_Free(pol2);
    
    return(0);
}  // end daisy_integrate

Mk_Doc(
    zero_select,
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    "\n +                   ds_zero_select                   +"
    "\n +  select integrated DSs with nearly zero velocity   +"
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    "\n     usage: daisy zero_select integrate.xyi 0.6         \n"
    "\n            integrate.xyi  -  integrated data file"
    "\n            0.6 (mm/year)  -  zero data criteria\n"
    "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

static int zero_select(int argc, char **argv)
{
    uint n = 0, nz = 0, nt = 0;

    const char *inp,   // input file
               *out1,  // output file - zero velocity
               *out2;  // output file - non-zero velocity

    float zch;

    FILE *in, *ou1, *ou2;

    float lon, lat, he, ve, vu;
    
    Aux_CheckArg(zero_select, 4);
    
    inp  = argv[2];
    out1 = argv[3];
    out2 = argv[4];
    
    zch = (float) atof(argv[5]);
    
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +                   ds_zero_select                   +"
           "\n +  select integrated DSs with nearly zero velocity   +"
           "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    printf("\n    input: %s"  , inp);
    println("\n   output: %s\n           %s", out1, out2);
    println("\n   zero DSs < |%3.1f| mm/year", zch);

    in = sfopen(inp, "rt");
    ou1 = sfopen(out1, "w+t");
    ou2 = sfopen(out2, "w+t");

//---------------------------------------------------------------

    while (fscanf(in, "%f %f %f %f %f", &lon, &lat, &he, &ve, &vu) > 0) {
        if (sqrt(ve * ve + vu * vu) <= zch) {
            fprintf(ou1, "%16.7e %15.7e %9.3f %7.3f %7.3f\n",
                        lon, lat, he, ve, vu);
            nz++;
        }
        else {
            fprintf(ou2, "%16.7e %15.7e %9.3f %7.3f %7.3f\n",
                        lon, lat, he, ve, vu);
            nt++;
        }
        n++; if(!(n % 1000)) printf("\n %6d ...", n);
    }
    printf("\n %6d\n", n);

    printf("\n   zero dominant DSs %6d\n        target   DSs %6d\n",nz,nt);
    printf("\n Records of output files:\n");
    printf("\n longitude latitude  height  ew_v   up_v");
    printf("\n (     degree          m       mm/year )\n");

//----------------------------------------------------------------------

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"
           "\n +              end    ds_zero_select                 +"
           "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

    return(0);

}
// end zero_select


/*
// Function for testing stuff
static PyObject * test(PyObject * self, PyObject * args)
{
    PyObject * arg = NULL;
    NPY_AO * array = NULL;
    
    if (!PyArg_ParseTuple(args, "O:test", &arg))
        return NULL;

    array = (NPY_AO *) PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if(array == NULL) goto fail;

    println("%ld %ld", PyArray_DIM(array, 0), PyArray_DIM(array, 1));
    println("%lf %lf", NPY_DELEM(array, 0, 0), NPY_DELEM(array, 0, 1));
    println("%lf %lf", NPY_DELEM(array, 1, 0), NPY_DELEM(array, 1, 1));
    
    Py_DECREF(array);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(array);
    return NULL;
}
*/

int main(int argc, char **argv)
{
    int ret;
    
    if (argc < 2) {
        exit(Err_Arg);
    }
    
    if (Str_Select("data_select"))
        return data_select(argc, argv);
    
    else if(Str_Select("dominant"))
        return dominant(argc, argv);
    
    else if(Str_Select("poly_orbit"))
        return poly_orbit(argc, argv);
    
    else if(Str_Select("integrate"))
        return integrate(argc, argv);
    else {
        errorln("Unrecognized module: %s", argv[1]);
        exit(Err_Arg);
    }
    
    return(0);
}
