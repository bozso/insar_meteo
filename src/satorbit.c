#include <stdio.h>
#include <tgmath.h>
#include <string.h>
#include <stdlib.h>

#include "insar.h"
#include "aux_macros.h"

#define Modules "azi_inc"
#define min_arg 2

typedef struct {
    uint is_centered, deg;
    double * coeffs;
    double mean_coords[3];
    double t_start, t_stop, t_mean;
} orbit_fit;


int read_fit(const char * path, orbit_fit * orb)
{
    FILE * fit_file = NULL;
    uint centered, deg;
    
    if ((fit_file = fopen(path, "r")) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("read_fit");
        return 1;
    }
    
    aux_read(fit_file, "centered: %d\n", &centered);
    
    if (centered) {
        aux_read(fit_file, "t_mean: %lf\n", &orb->t_mean);
        aux_read(fit_file, "mean_coords: %lf %lf %lf\n",
                 orb->mean_coords, orb->mean_coords + 1, orb->mean_coords + 2);
    }
    
    aux_read(fit_file, "deg: %d\n", &deg);
    aux_read(fit_file, "t_start: %lf\n", &orb->t_start);
    aux_read(fit_file, "t_stop: %lf\n", &orb->t_stop);
    
    orb->coeffs = aux_malloc(double, 3 * (deg + 1));

    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */

    aux_read(fit_file, "coeffs: %lf", orb->coeffs);
    
    FOR(ii, 1, 3 * (deg + 1))
        aux_read(fit_file, "%lf", orb->coeffs + ii);
    
    orb->is_centered = centered;
    orb->deg         = deg;
    
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
    
    uint n_poly = orb->deg + 1, is_centered = orb->is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (is_centered)
        time -= orb->t_mean;
    
    if(n_poly == 2) {
        x = coeffs[0] * time + coeffs[1];
        y = coeffs[2] * time + coeffs[3];
        z = coeffs[4] * time + coeffs[5];
    }
    else {
        x = coeffs[0]           * time;
        y = coeffs[n_poly]      * time;
        z = coeffs[2 * n_poly]  * time;

        FOR(ii, 1, n_poly - 1) {
            x = (x + coeffs[             ii]) * time;
            y = (y + coeffs[    n_poly + ii]) * time;
            z = (z + coeffs[2 * n_poly + ii]) * time;
        }
        
        x += coeffs[    n_poly - 1];
        y += coeffs[2 * n_poly - 1];
        z += coeffs[3 * n_poly - 1];
    }
    
    if (is_centered) {
        x += mean_coords[0];
        y += mean_coords[1];
        z += mean_coords[2];
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
    uint n_poly = orb->deg + 1;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (orb->is_centered)
        time -= orb->t_mean;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs[0] * time + coeffs[1];
        sat_y = coeffs[2] * time + coeffs[3];
        sat_z = coeffs[4] * time + coeffs[5];
        vel_x = coeffs[0]; vel_y = coeffs[2]; vel_z = coeffs[4];
    }
    // evaluation of polynom with Horner's method
    else {
        
        sat_x = coeffs[0]           * time;
        sat_y = coeffs[n_poly]      * time;
        sat_z = coeffs[2 * n_poly]  * time;

        FOR(ii, 1, n_poly - 1) {
            sat_x = (sat_x + coeffs[             ii]) * time;
            sat_y = (sat_y + coeffs[    n_poly + ii]) * time;
            sat_z = (sat_z + coeffs[2 * n_poly + ii]) * time;
        }
        
        sat_x += coeffs[    n_poly - 1];
        sat_y += coeffs[2 * n_poly - 1];
        sat_z += coeffs[3 * n_poly - 1];
        
        vel_x = coeffs[    n_poly - 2];
        vel_y = coeffs[2 * n_poly - 2];
        vel_z = coeffs[3 * n_poly - 2];
        
        FOR(ii, 0, n_poly - 3) {
            power = (double) n_poly - 1.0 - ii;
            vel_x += ii * coeffs[             ii] * pow(time, power);
            vel_y += ii * coeffs[    n_poly + ii] * pow(time, power);
            vel_z += ii * coeffs[2 * n_poly + ii] * pow(time, power);
        }
    }
    
    if (orb->is_centered) {
        sat_x += mean_coords[0];
        sat_y += mean_coords[1];
        sat_z += mean_coords[2];
    }
    
    // satellite coordinates - GNSS coordinates
    dx = sat_x - X;
    dy = sat_y - Y;
    dz = sat_z - Z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return((vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm);
}
// end dot_product

static void closest_appr(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, const uint max_iter, cart * sat_pos)
{
    // Compute the sat position using closest approche.
    
    // first, last and middle time, extending the time window by 5 seconds
    double t_start = orb->t_start - 5.0,
           t_stop  = orb->t_stop + 5.0,
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
" Usage: satorbit azi_inc fit_file infile mode max_iter outfile\
\n \
\n fit_file - ascii file with fit parameters\
\n infile   - inputfile with coordinates\
\n mode     - xyz for WGS-84 coordinates, llh for WGS-84 lon., lat., height\
\n max_iter - maximum number of iterations when calculating closest approache\
\n outfile  - binary output will be printed to this file\
\n");

int azi_inc(int argc, char **argv)
{
    aux_checkarg(azi_inc, 5);
    
    FILE *infile, *outfile;
    uint is_lonlat, max_iter = atoi(argv[5]), ndata = 0;
    cart sat, coord_xyz;
    llh coord_llh;
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
        while (fread(&coord_llh, sizeof(llh), 1, infile) > 0) {
            lon = coord_llh.lon * DEG2RAD;
            lat = coord_llh.lat * DEG2RAD;
            h   = coord_llh.h;
            
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
        while (fread(&coord_xyz, sizeof(cart), 1, infile) > 0) {
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &coord_xyz.x, &coord_xyz.y, &coord_xyz.z);
            
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
