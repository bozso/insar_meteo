#include <stdio.h>
#include <tgmath.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_matrix_double.h>

#include "aux_macros.h"

#define Modules "azi_inc"
#define min_arg 2

#define norm(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

typedef unsigned int uint;
typedef const double cdouble;

typedef struct { double x, y, z; } cart; // Cartesian coordinates
typedef struct { double lon, lat, h; } llh; // Cartesian coordinates

typedef struct {
    uint deg;
    double * coeffs;
} orbit_fit;

/************************
 * Auxilliary functions *
 * **********************/

static void * malloc_or_exit(size_t nbytes, const char * file, int line)
{
    /* Extended malloc function. */	
    void *x;
	
    if ((x = malloc(nbytes)) == NULL) {
        errorln("%s:line %d: malloc() of %zu bytes failed",
                file, line, nbytes);
        exit(err_alloc);
    }
    else
        return x;
}

static int read_fit(const char * path, orbit_fit * orb)
{
    FILE * fit_file = NULL;
    uint deg;
    
    if ((fit_file = fopen(path, "r")) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("read_fit");
        return 1;
    }
    aux_read(fit_file, "deg: %d\n", &deg);
    orb->coeffs = aux_malloc(double, 3 * (deg + 1));

    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */

    FOR(ii, 0, 3 * (deg + 1))
        aux_read(fit_file, "%lf", orb->coeffs + ii);
    
    orb->deg = deg;
    
    fclose(fit_file);
    return 0;
}

static void ell_cart (cdouble lon, cdouble lat, cdouble h,
                      double *x, double *y, double *z)
{
    /* From ellipsoidal to cartesian coordinates. */
    
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));;

    *x = (              n + h) * cos(lat) * cos(lon);
    *y = (              n + h) * cos(lat) * sin(lon);
    *z = ( (1.0 - E2) * n + h) * sin(lat);

} // end of ell_cart

static void cart_ell (cdouble x, cdouble y, cdouble z,
                      double *lon, double *lat, double *h)
{
    /* From cartesian to ellipsoidal coordinates. */
    
    double n, p, o, so, co;

    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    *lat = o;
    
    o = atan(y/x); if(x < 0.0) o += M_PI;
    *lon = o;
    *h = p / co - n;
}
// end of cart_ell

static void calc_pos(const orbit_fit * orb, double time, cart * pos)
{
    /* Calculate satellite position based on fitted polynomial orbits
     * at `time`. */
    
    uint n_poly = orb->deg + 1, deg = orb->deg;
    double x = 0.0, y = 0.0, z = 0.0;
    
    cdouble *coeffs = orb->coeffs;
    
    if(n_poly == 2) {
        x = coeffs[0] + coeffs[1] * time;
        y = coeffs[2] + coeffs[3] * time;
        z = coeffs[4] + coeffs[5] * time;
    }
    else {
        // highest degree
        x = coeffs[           + deg] * time;
        y = coeffs[    n_poly + deg] * time;
        z = coeffs[2 * n_poly + deg] * time;
        
        for(uint ii = deg - 1; ii >= 1; ii--) {
            x = (x + coeffs[             ii]) * time;
            y = (y + coeffs[    n_poly + ii]) * time;
            z = (z + coeffs[2 * n_poly + ii]) * time;
        }
        
        // lowest degree
        x += coeffs[         0];
        y += coeffs[    n_poly];
        z += coeffs[2 * n_poly];
    }
    
    pos->x = x; pos->y = y; pos->z = z;
} // end calc_pos

static double dot_product(const orbit_fit * orb, cdouble X, cdouble Y,
                          cdouble Z, double time)
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb->deg + 1, deg = orb->deg;
    
    const double *coeffs = orb->coeffs;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs[0] + coeffs[1] * time;
        sat_y = coeffs[2] + coeffs[3] * time;
        sat_z = coeffs[4] + coeffs[5] * time;
        
        vel_x = coeffs[1]; vel_y = coeffs[3]; vel_z = coeffs[5];
    }
    // evaluation of polynom with Horner's method
    else {
        // highest degree
        sat_x = coeffs[           + deg] * time;
        sat_y = coeffs[    n_poly + deg] * time;
        sat_z = coeffs[2 * n_poly + deg] * time;

        for(uint ii = deg - 1; ii >= 1; ii--) {
            sat_x = (sat_x + coeffs[             ii]) * time;
            sat_y = (sat_y + coeffs[    n_poly + ii]) * time;
            sat_z = (sat_z + coeffs[2 * n_poly + ii]) * time;
        }
        
        // lowest degree
        sat_x += coeffs[         0];
        sat_y += coeffs[    n_poly];
        sat_z += coeffs[2 * n_poly];
        
        // constant term
        vel_x = coeffs[1             ];
        vel_y = coeffs[1 +     n_poly];
        vel_z = coeffs[1 + 2 * n_poly];
        
        // linear term
        vel_x += coeffs[2             ] * time;
        vel_y += coeffs[2 +     n_poly] * time;
        vel_z += coeffs[2 + 2 * n_poly] * time;
        
        FOR(ii, 3, n_poly) {
            power = (double) ii - 1;
            vel_x += ii * coeffs[             ii] * pow(time, power);
            vel_y += ii * coeffs[    n_poly + ii] * pow(time, power);
            vel_z += ii * coeffs[2 * n_poly + ii] * pow(time, power);
        }
    }
    // satellite coordinates - GNSS coordinates
    dx = sat_x - X;
    dy = sat_y - Y;
    dz = sat_z - Z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return (vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm;
}
// end dot_product

static void closest_appr(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, const uint max_iter, cart * sat_pos)
{
    /* Compute the sat position using closest approche. */
    
    // first, last and middle time
    // domain of t is [-1.0, 1.0]
    double t_start = -1.0000001,
           t_stop  =  1.0000001,
           t_middle;
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, X, Y, Z, t_middle);
        
        // change start for middle
        if ((dot_start * dot_middle) > 0.0) {
            t_start = t_middle;
            dot_start = dot_middle;
        }
        // change  end  for middle
        else
            t_stop = t_middle;

        itr++;
    }
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // end closest_appr

/***********************************************
 * Main functions - calleble from command line *
 ***********************************************/

mk_doc(azi_inc,
"\n Usage: inmet_utils azi_inc fit_file coords mode max_iter outfile\
 \n \
 \n fit_file - ascii file with fit parameters\
 \n coords   - inputfile with coordinates\
 \n mode     - xyz for WGS-84 coordinates, llh for WGS-84 lon., lat., height\
 \n max_iter - maximum number of iterations when calculating closest approache\
 \n outfile  - binary output will be printed to this file\
 \n\n");

int azi_inc(int argc, char **argv)
{
    aux_checkarg(azi_inc, 5);
    
    FILE *infile, *outfile;
    uint is_lonlat, max_iter = atoi(argv[5]), ndata = 0;
    cart sat;
    double coords[3];
    orbit_fit orb;
    
    // topocentric parameters in PS local system
    double xf, yf, zf,
           xl, yl, zl,
           X, Y, Z,
           t0, lon, lat, h,
           azi, inc;
    
    if (read_fit(argv[2], &orb)) {
        errorln("Could not read orbit fit file %s. Exiting!", argv[2]);
        return err_io;
    }
    
    aux_fopen(infile, argv[3], "rb");
    aux_fopen(outfile, argv[6], "wb");
    
    // infile contains lon, lat, h
    if (str_isequal(argv[4], "llh")) {
        while (fread(coords, sizeof(double), 3, infile) > 0) {
            lon = coords[0] * DEG2RAD;
            lat = coords[1] * DEG2RAD;
            h   = coords[2];
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            // satellite closest approache cooridantes
            closest_appr(&orb, X, Y, Z, max_iter, &sat);
            
            xf = sat.x - X;
            yf = sat.y - Y;
            zf = sat.z - Z;
            
            // estiamtion of azimuth and inclination
            xl = - sin(lat) * cos(lon) * xf
                 - sin(lat) * sin(lon) * yf + cos(lat) * zf ;
        
            yl = - sin(lon) * xf + cos(lon) * yf;
        
            zl = + cos(lat) * cos(lon) * xf
                 + cos(lat) * sin(lon) * yf + sin(lat) * zf ;
        
            t0 = norm(xl, yl, zl);
        
            inc = acos(zl / t0) * RAD2DEG;
        
            if(xl == 0.0) xl = 0.000000001;
        
            azi = atan(fabs(yl / xl));
        
            if( (xl < 0.0) && (yl > 0.0) ) azi = M_PI - azi;
            if( (xl < 0.0) && (yl < 0.0) ) azi = M_PI + azi;
            if( (xl > 0.0) && (yl < 0.0) ) azi = 2.0 * M_PI - azi;
        
            azi *= RAD2DEG;
        
            if(azi > 180.0)
                azi -= 180.0;
            else
                azi += 180.0;
            
            fwrite(&azi, sizeof(double), 1, outfile);
            fwrite(&inc, sizeof(double), 1, outfile);
            ndata++;
            
            if (!(ndata % 1000))
                println("Processed %d points.", ndata);
        } // end while
    }
    // infile contains X, Y, Z
    else if (str_isequal(argv[4], "xyz")) {
        while (fread(coords, sizeof(double), 3, infile) > 0) {
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &coords[0], &coords[1], &coords[2]);
            
            // satellite closest approache cooridantes
            closest_appr(&orb, X, Y, Z, max_iter, &sat);
            
            xf = sat.x - X;
            yf = sat.y - Y;
            zf = sat.z - Z;
            
            // estiamtion of azimuth and inclination
            xl = - sin(lat) * cos(lon) * xf
                 - sin(lat) * sin(lon) * yf + cos(lat) * zf ;
        
            yl = - sin(lon) * xf + cos(lon) * yf;
        
            zl = + cos(lat) * cos(lon) * xf
                 + cos(lat) * sin(lon) * yf + sin(lat) * zf ;
        
            t0 = norm(xl, yl, zl);
        
            inc = acos(zl / t0) * RAD2DEG;
        
            if(xl == 0.0) xl = 0.000000001;
        
            azi = atan(fabs(yl / xl));
        
            if( (xl < 0.0) && (yl > 0.0) ) azi = M_PI - azi;
            if( (xl < 0.0) && (yl < 0.0) ) azi = M_PI + azi;
            if( (xl > 0.0) && (yl < 0.0) ) azi = 2.0 * M_PI - azi;
        
            azi *= RAD2DEG;
        
            if(azi > 180.0)
                azi -= 180.0;
            else
                azi +=180.0;
            
            fwrite(&azi, sizeof(double), 1, outfile);
            fwrite(&inc, sizeof(double), 1, outfile);
            ndata++;
            
            if (!(ndata % 1000))
                println("Processed %d points.", ndata);

        } // end while
    } // end else if
    else {
        errorln("Third argument should be either llh or xyz not %s!",
                argv[4]);
        return err_arg;
    }
    fclose(infile);
    fclose(outfile);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        errorln("At least one argument (module name) is required.\
                 \nModules to choose from: %s.", Modules);
        printf("Use --help or -h as the first argument to print the help message.\n");
        return err_arg;
    }
    
    if (module_select("azi_inc") || module_select("AZI_INC"))
        return azi_inc(argc, argv);
    
    //else if (Module_Select("dominant") || Module_Select("DOMINANT"))
    //    return dominant(argc, argv);

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return err_arg;
    }
    
    return 0;
}
