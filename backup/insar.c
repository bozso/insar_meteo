#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include "aux_macros.h"

/* Iterating over array   
 * for(uint jj = 0; jj < ncols; jj++)
 *      for(uint ii = 0; ii < nrows; ii++)
 *          data[jj * nrows + ii] = ...;
 */

#define Idx(ii, jj, nrows) (ii) * (nrows) + (jj)

typedef unsigned int uint;
typedef const double cdouble;

//-----------------------------------------------------------------------------
// STRUCTS
//-----------------------------------------------------------------------------

typedef struct { float lon, lat;  } psxy;
typedef struct { int ni; float lon, lat, he, ve; } psxys;
typedef struct { double x, y, z, lat, lon, h; } station; // [m,rad]

typedef struct { double x, y, z; } cart; // Cartesian coordinates
typedef struct { double lon, lat, h; } llh; // Cartesian coordinates

typedef struct {
    uint is_centered, deg;
    double * coeffs, *mean_coords;
    double t_start, t_stop, t_mean;
} orbit_fit;

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

static void ell_cart (const llh *lonlat, cart * coord)
{
    // from ellipsoidal to cartesian coordinates

    double lat, lon, h, n;

    lat = lonlat->lat;
    lon = lonlat->lon;
    h   = lonlat->h;
    n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    coord->x = (              n + h) * cos(lat) * cos(lon);
    coord->y = (              n + h) * cos(lat) * sin(lon);
    coord->z = ( (1.0 - E2) * n + h) * sin(lat);

}
// end of ell_cart

static void cart_ell (const cart * coord, llh * lonlat)
{
    // from cartesian to ellipsoidal coordinates

    double n, p, o, so, co, x, y, z;

    n = (WA * WA - WB * WB);
    x = coord->x; y = coord->y; z = coord->z;
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    lonlat->lat = o;
    
    o = atan(y/x); if(x < 0.0) o += M_PI;
    lonlat->lon = o;
    lonlat->h = p / co - n;
}
// end of cart_ell

static double norm(cdouble x, cdouble y, cdouble z)
{
    return sqrt(x * x + y * y + z * z);
}

/* ---------------------------------------------------------------------------
 * Fitting polynomial to orbit, closest approache and azimuth, incidence angle
 * calculation.
 * --------------------------------------------------------------------------*/

static void calc_pos(const orbit_fit * orb, double time, cart * pos)
{
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
} // calc_pos

static double dot_product(const orbit_fit * orb, const cart * coord,
                          double time)
{
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
    
    // satellite coordinates - surface coordinates
    dx = sat_x - coord->x;
    dy = sat_y - coord->y;
    dz = sat_z - coord->z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return(vel_x * dx * inorm + vel_y * dy * inorm + vel_z * dz * inorm);
}

static void closest_appr(const orbit_fit * orb, const cart * coord,
                         const uint max_iter, cart * sat_pos)
{
    // compute the sat position using closest approache
    
    // first, last and middle time
    double t_start = orb->t_start - 5.0,
           t_stop  = orb->t_stop + 5.0,
           t_middle; 
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, coord, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, coord, t_middle);
        
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
    
    calc_pos(orb, t_middle, sat_pos);
} // end closest_appr

void azi_inc(cdouble t_start, cdouble t_stop, cdouble t_mean,
             cdouble * coeffs, cdouble * coords, cdouble * mean_coords,
             double * azi_inc, uint ndata, uint is_centered, uint deg,
             uint max_iter, uint is_lonlat)
{
    // topocentric parameters in PS local system
    // double xf, yf, zf, xl, yl, zl, t0, lon, lat, azi, inc;
    
    println("%lf %lf %lf\n", coeffs[0], coeffs[1], coeffs[2]);
    
    //return;
    /*
    const orbit_fit orbit;
    
    orbit.t_start = t_start
    
    cart sat;
    const cart *coord;
    llh lonlat;
    
    FOR(ii, 0, ndata) {
        coord = coords + ii;
        closest_appr(orbit, coord, max_iter, &sat);
            
        xf = sat.x - coord->x;
        yf = sat.y - coord->y;
        zf = sat.z - coord->z;
        
        cart_ell(coord, &lonlat);
        
        lon = lonlat->lon;
        lat = lonlat->lat;
        
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
        
        azimuths[ii] = azi;
        inclinations[ii] = inc;
    }
    * */
}

/*
void poly_sat_vel(double *vx, double *vy, double *vz, const double time,
                  const double *poli, const uint deg_poly)
{
    *vx = 0.0;
    *vy = 0.0;
    *vz = 0.0;

    FOR(ii, 1, deg_poly) {
        *vx += ii * poli[ii]                * pow(time, (double) ii - 1);
        *vy += ii * poli[deg_poly + ii]     * pow(time, (double) ii - 1);
        *vz += ii * poli[2 * deg_poly + ii] * pow(time, (double) ii - 1);
    }
}
*/

/*
int fit_orbit(const torb * orbits, const uint ndata, const uint deg,
              const uint is_centered, const char * outfile)
{
    double * times,         // vector for times
             t_mean = 0.0,  // mean value of times
             t, x, y, z,    // temp storage variables
             x_mean = 0.0,  // x, y, z mean values
             y_mean = 0.0,
             z_mean = 0.0;
    
    int status;
    
    // vector for orbit coordinates
    gsl_vector *obs_x  = gsl_vector_alloc(ndata),
               *obs_y  = gsl_vector_alloc(ndata),
               *obs_z  = gsl_vector_alloc(ndata),
               *coeffs = gsl_vector_alloc(deg + 1);
    
    // design matrix
    gsl_matrix * design = gsl_matrix_alloc(ndata, deg);
    
    times = Aux_Malloc(double, ndata);
    
    if (is_centered) {
        FOR(ii, 0, ndata) {
            t = orbits[ii].t;
            times[ii] = t;
            
            t_mean += t;            
            
            x = orbits[ii].x;
            y = orbits[ii].y;
            z = orbits[ii].z;
            
            Vset(obs_x, ii, x);
            Vset(obs_y, ii, y);
            Vset(obs_z, ii, z);
            
            x_mean += x;
            y_mean += y;
            z_mean += z;
        }
        // calculate means
        t_mean /= (double) ndata;

        x_mean /= (double) ndata;
        y_mean /= (double) ndata;
        z_mean /= (double) ndata;
        
        // subtract mean value
        FOR(ii, 0, ndata) {
            times[ii] -= t_mean;

            *Vptr(obs_x, ii) -= x_mean;
            *Vptr(obs_y, ii) -= y_mean;
            *Vptr(obs_z, ii) -= z_mean;
        }
    }
    else {
        FOR(ii, 0, ndata) {
            times[ii] = orbits[ii].t;
            
            Vset(obs_x, ii, orbits[ii].x);
            Vset(obs_y, ii, orbits[ii].y);
            Vset(obs_z, ii, orbits[ii].z);
        }
    }
    
    FOR(ii, 0, ndata)
        Mset(design, ii, 0, 1.0);
    
    FOR(ii, 0, ndata)
        FOR(jj, 1, deg)
            Mset(design, ii, jj, times[ii] * Mget(design, ii, jj - 1));
    
    status = gsl_linalg_cholesky_decomp1(design);
    
    status = gsl_linalg_cholesky_solve(design, obs_x, coeffs);
    status = gsl_linalg_cholesky_solve(design, obs_y, coeffs);
    status = gsl_linalg_cholesky_solve(design, obs_z, coeffs);
    
    gsl_vector_free(obs_x);
    gsl_vector_free(obs_y);
    gsl_vector_free(obs_z);

    gsl_vector_free(coeffs);

    gsl_matrix_free(design);
    
    return status;
}
// end fit_orbit
*/
static void axd(cdouble  a1, cdouble  a2, cdouble  a3,
                cdouble  d1, cdouble  d2, cdouble  d3,
                double *n1, double *n2, double *n3)
{
    // vectorial multiplication a x d
   *n1 = a2 * d3 - a3 * d2;
   *n2 = a3 * d1 - a1 * d3;
   *n3 = a1 * d2 - a2 * d1;
}

void testfun(double * data, uint nrows, uint ncols)
{
    FOR(jj, 0, ncols)
        FOR(ii, 0, nrows)
            data[Idx(ii, jj, nrows)] = jj;
}

