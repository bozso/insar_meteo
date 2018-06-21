#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>

#include "aux_macros.h"

#define Modules "azi_inc fit_orbit"
#define min_arg 2

#define norm(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

#define BUFSIZE 10

typedef unsigned int uint;
typedef const double cdouble;

// Cartesian coordinates
typedef struct cart_struct { double x, y, z; } cart; 

// WGS-84 surface coordinates
typedef struct llh_struct { double lon, lat, h; } llh; 

// Orbit records
typedef struct orbit_struct { double t, x, y, z; } orbit;

// Fitted orbit polynoms structure
typedef struct orbit_fit_struct {
    uint deg, centered;
    double t_min, t_max, t_mean;
    double *coeffs, coords_mean[3];
} orbit_fit;

/************************
 * Auxilliary functions *
 * **********************/

static int read_fit(const char * path, orbit_fit * orb)
{
    FILE * fit_file = NULL;
    uint deg, centered;
    
    aux_open(fit_file, path, "r");
    
    aux_read(fit_file, "centered: %u\n", &centered);
    
    if (centered) {
        aux_read(fit_file, "t_mean: %lf\n", &(orb->t_mean));
        aux_read(fit_file, "coords_mean: %lf %lf %lf\n",
                                        &(orb->coords_mean[0]),
                                        &(orb->coords_mean[1]),
                                        &(orb->coords_mean[2]));
    } else {
        orb->t_mean = 0.0;
        orb->coords_mean[0] = 0.0;
        orb->coords_mean[1] = 0.0;
        orb->coords_mean[2] = 0.0;
    }
    
    aux_read(fit_file, "t_min: %lf\n", &orb->t_min);
    aux_read(fit_file, "t_max: %lf\n", &orb->t_max);
    aux_read(fit_file, "deg: %u\ncoeffs: ", &deg);

    aux_malloc(orb->coeffs, double, 3 * (deg + 1));

    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */    
    FOR(ii, 0, 3)
        FOR(jj, 0, deg + 1)
            aux_read(fit_file, "%lf ", orb->coeffs + ii);
    
    orb->deg = deg;
    orb->centered = centered;
    
    fclose(fit_file);
    return 0;
fail:
    aux_close(fit_file);
    aux_free(orb->coeffs);
    return 1;
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
    
    uint n_poly   = orb->deg + 1,
         deg      = orb->deg;
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
    
    if (orb->centered) {
        pos->x = x + orb->coords_mean[0];
        pos->y = y + orb->coords_mean[1];
        pos->z = z + orb->coords_mean[2];
    }
    else {
        pos->x = x;
        pos->y = y;
        pos->z = z;
    }
} // end calc_pos

static double dot_product(const orbit_fit * orb, cdouble X, cdouble Y,
                          cdouble Z, double time)
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb->deg + 1,
         deg = orb->deg;
    
    cdouble *coeffs = orb->coeffs;
    
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

    if (orb->centered) {
        sat_x += orb->coords_mean[0];
        sat_y += orb->coords_mean[1];
        sat_z += orb->coords_mean[2];
    }

    // satellite coordinates - GNSS coordinates
    dx = sat_x - X;
    dy = sat_y - Y;
    dz = sat_z - Z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    // scalar product of delta vector and velocities
    return (vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm;
}
// end dot_product

static void closest_appr(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, const uint max_iter, cart * sat_pos)
{
    /* Compute the sat position using closest approche. */
    
    // first, last and middle time
    double t_min = orb->t_min,
           t_max = orb->t_max,
           t_middle;
    
    if (orb->centered) {
        t_min -= orb->t_mean;
        t_max -= orb->t_mean;
    }
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_min);
    
    while (fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
        t_middle = (t_min + t_max) / 2.0;

        dot_middle = dot_product(orb, X, Y, Z, t_middle);
        
        // change start for middle
        if ((dot_start * dot_middle) > 0.0) {
            t_min = t_middle;
            dot_start = dot_middle;
        }
        // change  end  for middle
        else
            t_max = t_middle;

        itr++;
    }
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // end closest_appr

/***********************************************
 * Main functions - calleble from command line *
 ***********************************************/

mk_doc(fit_orbit,
"\n Usage: inmet fit_orbit coords deg is_centered fit_file\
 \n \
 \n coords      - ascii file with (t,x,y,z) coordinates\
 \n deg         - degree of fitted polynom\
 \n is_centered - 1 to subtract mean time and coordinates from time points and \
 \n               coordinates\
 \n fit_file    - ascii fit output will be written into this file\
 \n\n");

int fit_orbit(int argc, char **argv)
{
    aux_checkarg(fit_orbit, 4);

    FILE *incoords, *fit_file;
    uint deg = (uint) atoi(argv[3]);
    uint is_centered = (uint) atoi(argv[4]);
    uint idx = 0, ndata = 0;
    uint max_idx = BUFSIZE - 1;
    
    double residual[] = {0.0, 0.0, 0.0};
    
    aux_open(incoords, argv[2], "r");
    
    orbit * orbits;
    
    aux_malloc(orbits, orbit, BUFSIZE);
    
    double t_mean = 0.0,  // mean value of times
           t, x, y, z,    // temp storage variables
           x_mean = 0.0,  // x, y, z mean values
           y_mean = 0.0,
           z_mean = 0.0,
           t_min, t_max, res_tmp;
    
    gsl_vector *tau, // vector for QR decompisition
               *res; // vector for holding residual values
    
    // matrices
    gsl_matrix *design, *obs, *fit;
    
    // vector views of matrix columns and rows
    gsl_vector_view fit_view;
    
    if (is_centered) {
        while(fscanf(incoords, "%lf %lf %lf %lf\n", &t, &x, &y, &z) > 0) {
            ndata++;
            
            t_mean += t;
            x_mean += x;
            y_mean += y;
            z_mean += z;
            
            orbits[idx].t = t;
            orbits[idx].x = x;
            orbits[idx].y = y;
            orbits[idx].z = z;

            idx++;
            
            if (idx >= max_idx) {
                aux_realloc(orbits, orbit, 2 * idx);
                max_idx = 2 * idx - 1;
            }
        }

        // calculate means
        t_mean /= (double) ndata;
    
        x_mean /= (double) ndata;
        y_mean /= (double) ndata;
        z_mean /= (double) ndata;
    }
    else {
        while(fscanf(incoords, "%lf %lf %lf %lf\n",
                     &orbits[idx].t, &orbits[idx].x, &orbits[idx].y,
                     &orbits[idx].z) > 0) {
            idx++;
            ndata++;
            
            if (idx >= max_idx) {
                aux_realloc(orbits, orbit, 2 * idx);
                max_idx = 2 * idx - 1;
            }
        }
    }
    
    t_min = orbits[0].t;
    
    FOR(ii, 1, ndata) {
        t = orbits[ii].t;
        
        if (t < t_min)
            t_min = t;
    }

    t_max = orbits[0].t;
    
    FOR(ii, 1, ndata) {
        t = orbits[ii].t;
        
        if (t > t_max)
            t_max = t;
    }
    
    if (ndata < (deg + 1)) {
        errorln("Underdetermined system, we have less data points (%d) than\
                 \nunknowns (%d)!", ndata, deg + 1);
        goto fail;
    }

    obs = gsl_matrix_alloc(ndata, 3);
    fit = gsl_matrix_alloc(3, deg + 1);

    design = gsl_matrix_alloc(ndata, deg + 1);
    
    tau = gsl_vector_alloc(deg + 1);
    res = gsl_vector_alloc(ndata);
    
    FOR(ii, 0, ndata) {
        // fill up matrix that contains coordinate values
        Mset(obs, ii, 0, orbits[ii].x - x_mean);
        Mset(obs, ii, 1, orbits[ii].y - y_mean);
        Mset(obs, ii, 2, orbits[ii].z - z_mean);
        
        t = orbits[ii].t - t_mean;
        
        // fill up design matrix
        
        // first column is ones
        Mset(design, ii, 0, 1.0);
        
        // second column is t values
        Mset(design, ii, 1, t);
        
        // further columns contain the power of t values
        FOR(jj, 2, deg + 1)
            *Mptr(design, ii, jj) = Mget(design, ii, jj - 1) * t;
    }
    
    // factorize design matrix
    if (gsl_linalg_QR_decomp(design, tau)) {
        error("QR decomposition failed.\n");
        goto fail;
    }
    
    // do the fit for x, y, z
    FOR(ii, 0, 3) {
        fit_view = gsl_matrix_row(fit, ii);
        gsl_vector_const_view coord = gsl_matrix_const_column(obs, ii);
        
        if (gsl_linalg_QR_lssolve(design, tau, &coord.vector, &fit_view.vector,
                                  res)) {
            error("Solving of linear system failed!\n");
            goto fail;
        }
        
        res_tmp = 0.0;
        
        // calculate RMS of residual values
        FOR(jj, 0, ndata)
            res_tmp += Vget(res, jj) * Vget(res, jj);
        
        residual[ii] = sqrt(res_tmp / ndata);
    }
    
    aux_open(fit_file, argv[5], "w");
    
    fprintf(fit_file, "centered: %u\n", is_centered);
    
    if (is_centered) {
        fprintf(fit_file, "t_mean: %lf\n", t_mean);
        fprintf(fit_file, "coords_mean: %lf %lf %lf\n",
                                        x_mean, y_mean, z_mean);
    }
    
    fprintf(fit_file, "t_min: %lf\n", t_min);
    fprintf(fit_file, "t_max: %lf\n", t_max);
    fprintf(fit_file, "deg: %u\n", deg);
    fprintf(fit_file, "coeffs: ");
    
    FOR(ii, 0, 3)
        FOR(jj, 0, deg + 1)
            fprintf(fit_file, "%lf ", Mget(fit, ii, jj));

    fprintf(fit_file, "\nRMS of residuals (x, y, z) [m]: (%lf, %lf, %lf)\n",
                      residual[0], residual[1], residual[2]);
    
    fprintf(fit_file, "\n");
    
    gsl_matrix_free(design);
    gsl_matrix_free(obs);
    gsl_matrix_free(fit);
    
    gsl_vector_free(tau);
    gsl_vector_free(res);
    
    aux_free(orbits);
    
    fclose(incoords);
    fclose(fit_file);
    
    return 0;

fail:
    gsl_matrix_free(design);
    gsl_matrix_free(obs);
    gsl_matrix_free(fit);
    
    gsl_vector_free(tau);
    gsl_vector_free(res);
    
    aux_free(orbits);
    
    aux_close(incoords);
    aux_close(fit_file);
    
    return 1;
}

mk_doc(eval_orbit,
"\n Usage: inmet eval_orbit fit_file steps outfile\
 \n \
 \n fit_file    - ascii file that contains fitted orbit polynom parameters\
 \n nstep       - evaluate x, y, z coordinates at nstep number of steps\
 \n               between the range of t_min and t_max\
 \n outfile     - coordinates and time values will be written to this ascii file\
 \n\n");

int eval_orbit(int argc, char **argv)
{
    aux_checkarg(eval_orbit, 3);
    
    FILE *outfile;
    orbit_fit orb;
    uint nstep;
    double t_min, t_mean, dstep, t;
    cart pos;
    
    if (read_fit(argv[2], &orb)) {
        errorln("Could not read orbit fit file %s. Exiting!", argv[2]);
        return err_io;
    }
    
    t_min = orb.t_min;
    nstep = (uint) atoi(argv[3]);
    
    dstep = (orb.t_max - t_min) / (double) nstep;
    
    aux_open(outfile, argv[4], "w");
    
    if (orb.centered) {
        t_mean = orb.t_mean;
        
        FOR(ii, 0, nstep + 1) {
            t = t_min - t_mean + ii * dstep;
            calc_pos(&orb, t, &pos);
            fprintf(outfile, "%lf %lf %lf %lf\n", t + t_mean, pos.x, pos.y, pos.z);
        }
    } else {
        FOR(ii, 0, nstep + 1) {
            t = t_min + ii * dstep;
            calc_pos(&orb, t, &pos);
            fprintf(outfile, "%lf %lf %lf %lf\n", t, pos.x, pos.y, pos.z);
        }
    }
    
    fclose(outfile);
    return 0;

fail:    
    aux_close(outfile);
    return 1;
}

mk_doc(azi_inc,
"\n Usage: inmet azi_inc fit_file coords mode max_iter outfile\
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
    uint is_lonlat, max_iter = atoi(argv[5]);
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
    
    aux_open(infile, argv[3], "rb");
    aux_open(outfile, argv[6], "wb");
    
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
        } // end while
    } // end else if
    else {
        errorln("Third argument should be either llh or xyz not %s!",
                argv[4]);
        goto fail;
    }
    fclose(infile);
    fclose(outfile);
    return 0;

fail:
    aux_close(infile);
    aux_close(outfile);
    return 1;
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
    
    else if (module_select("fit_orbit") || module_select("FIT_ORBIT"))
        return fit_orbit(argc, argv);

    else if (module_select("eval_orbit") || module_select("EVAL_ORBIT"))
        return eval_orbit(argc, argv);

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return err_arg;
    }
    
    return 0;
}
