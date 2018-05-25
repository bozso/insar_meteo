#include <stdio.h>
#include <tgmath.h>
#include <stdlib.h>

#include "insar.h"
#include "aux_macros.h"

/* Iterating over array   
 * for(uint jj = 0; jj < ncols; jj++)
 *      for(uint ii = 0; ii < nrows; ii++)
 *          data[Idx(ii, jj, nrows)] = ...;
 */


/************************
 * Auxilliary functions *
 * **********************/

/* Extended malloc function */
static void * malloc_or_exit(size_t nbytes, const char * file, int line)
{	
    void *x;
	
    if ((x = malloc(nbytes)) == NULL) {
        errorln("%s:line %d: malloc() of %zu bytes failed",
                file, line, nbytes);
        exit(Err_Alloc);
    }
    else
        return x;
}

/* Safely open files. */
static FILE * sfopen(const char * path, const char * mode)
{
    FILE * file = fopen(path, mode);

    if (!file) {
        errorln("Error opening file: \"%s\". ", path);
        perror("fopen");
        exit(Err_Io);
    }
    return file;
}

static double norm(cdouble x, cdouble y, cdouble z)
{
    // vector norm
    return sqrt(x * x + y * y + z * z);
}


static void ell_cart (cdouble lon, cdouble lat, cdouble h,
                      double *x, double *y, double *z)
{
    // from ellipsoidal to cartesian coordinates
    
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));;

    *x = (              n + h) * cos(lat) * cos(lon);
    *y = (              n + h) * cos(lat) * sin(lon);
    *z = ( (1.0 - E2) * n + h) * sin(lat);

} // end of ell_cart


static void cart_ell (cdouble x, cdouble y, cdouble z,
                      double *lon, double *lat, double *h)
{
    // from cartesian to ellipsoidal coordinates
    
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
    // Calculate satellite position based on fitted polynomial orbits at time
    
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

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

EXPORTED_FUNCTION void azi_inc(cdouble start_t, cdouble stop_t, cdouble mean_t,
                               double * coeffs, double * coords,
                               double * mean_coords, double * azi_inc,
                               uint is_centered, uint deg, uint max_iter,
                               uint is_lonlat, uint ncoords)
{
    cart sat;
    orbit_fit orb;

    // topocentric parameters in PS local system
    double xf, yf, zf,
           xl, yl, zl,
           X, Y, Z,
           t0, lon, lat, h,
           azi, inc;
    
    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */

    // Set up orbit polynomial structure
    orb.coeffs = coeffs;
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.t_start = start_t;
    orb.t_stop = stop_t;
    orb.t_mean = mean_t;
    
    orb.mean_coords = mean_coords;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, ncoords) {
            lon = coords[Idx(ii, 0, ncoords)] * DEG2RAD;
            lat = coords[Idx(ii, 1, ncoords)] * DEG2RAD;
            h   = coords[Idx(ii, 2, ncoords)] * DEG2RAD;
            
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
        
            azi = atan(abs(yl / xl));
        
            if( (xl < 0.0) && (yl > 0.0) ) azi = M_PI - azi;
            if( (xl < 0.0) && (yl < 0.0) ) azi = M_PI + azi;
            if( (xl > 0.0) && (yl < 0.0) ) azi = 2.0 * M_PI - azi;
        
            azi *= RAD2DEG;
        
            if(azi > 180.0)
                azi -= 180.0;
            else
                azi +=180.0;
            
            azi_inc[Idx(ii, 0, ncoords)] = azi;
            azi_inc[Idx(ii, 1, ncoords)] = inc;
        }
        // end for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, ncoords) {
            X = coords[Idx(ii, 0, ncoords)];
            Y = coords[Idx(ii, 1, ncoords)];
            Z = coords[Idx(ii, 2, ncoords)];
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
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
        
            azi = atan(abs(yl / xl));
        
            if( (xl < 0.0) && (yl > 0.0) ) azi = M_PI - azi;
            if( (xl < 0.0) && (yl < 0.0) ) azi = M_PI + azi;
            if( (xl > 0.0) && (yl < 0.0) ) azi = 2.0 * M_PI - azi;
        
            azi *= RAD2DEG;
        
            if(azi > 180.0)
                azi -= 180.0;
            else
                azi +=180.0;
            
            azi_inc[Idx(ii, 0, ncoords)] = azi;
            azi_inc[Idx(ii, 1, ncoords)] = inc;
        }
        // end for
    }
    // end else
}
