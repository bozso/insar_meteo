/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <tgmath.h>
#include <iso646.h>

#include "satorbit.h"

static inline double norm(x, y, z)
    cdouble x, y, z;
{
    return sqrt(x * x + y * y + z * z);
}

// Calculate satellite position based on fitted polynomial orbits at time

static inline void calc_pos(orb, time, pos)
    const orbit_fit * orb;
    double time;
    cart * pos;
{
    uint n_poly = orb->deg + 1, is_centered = orb->is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (is_centered)
        time -= orb->mean_t;
    
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

static inline double dot_product(orb, X, Y, Z, time)
    const orbit_fit * orb;
    cdouble X, Y, Z;
    double time;
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb->deg + 1;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (orb->is_centered)
        time -= orb->mean_t;
    
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
            power = (double) (n_poly - 1.0 - ii);
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
    
    return (vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm;
} // dot_product

// Compute the sat position using closest approche.

static inline void closest_appr(orb, X, Y, Z, max_iter, sat_pos)
    const orbit_fit * orb;
    cdouble X, Y, Z;
    cuint max_iter;
    cart * sat_pos;
{
    // first, last and middle time, extending the time window by 5 seconds
    double t_start = orb->start_t - 5.0,
           t_stop  = orb->stop_t + 5.0,
           t_middle = 0.0;
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 and itr < max_iter) {
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
    } // while
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // closest_appr

void ell_cart (lon, lat, h, x, y, z)
    cdouble lon, lat, h;
    double *x, *y, *z;
{
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    *x = (              n + h) * cos(lat) * cos(lon);
    *y = (              n + h) * cos(lat) * sin(lon);
    *z = ( (1.0 - E2) * n + h) * sin(lat);

} // ell_cart

void cart_ell(x, y, z, lon, lat, h)
    cdouble x, y, z;
    double *lon, *lat, *h;
{
    double n, p, o, so, co;

    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    
    so = sin(o);
    co = cos(o);
    
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    
    so = sin(o);
    co = cos(o);
    
    n = WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    *lat = o;
    
    o = atan(y/x);
    
    if(x < 0.0)
        o += M_PI;
    
    *lon = o;
    *h = p / co - n;
} // cart_ell


void calc_azi_inc(orb, X, Y, Z, lon, lat, max_iter, azi, inc)
    const orbit_fit * orb;
    cdouble X, Y, Z, lon, lat;
    cuint max_iter;
    double *azi, *inc;
{
    double xf, yf, zf, xl, yl, zl, t0;
    cart sat;
    
    // satellite closest approache cooridantes
    closest_appr(orb, X, Y, Z, max_iter, sat);
    
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
    
    *inc = acos(zl / t0) * RAD2DEG;
    
    if(xl == 0.0) xl = 0.000000001;
    
    double temp_azi = atan(fabs(yl / xl));
    
    if( (xl < 0.0) && (yl > 0.0) ) temp_azi = M_PI - temp_azi;
    if( (xl < 0.0) && (yl < 0.0) ) temp_azi = M_PI + temp_azi;
    if( (xl > 0.0) && (yl < 0.0) ) temp_azi = 2.0 * M_PI - temp_azi;
    
    temp_azi *= RAD2DEG;
    
    if(temp_azi > 180.0)
        temp_azi -= 180.0;
    else
        temp_azi += 180.0;
    
    *azi = temp_azi;
} // calc_azi_inc

#if 0

void asc_dsc_select(double * _arr1, double * _arr2, double max_sep,
                    int rows1, int rows2, int cols, int * _idx, int nfound)
{
    nfound = 0;
    double dlon = 0, dlat = 0;
    
    array2d arr1(_arr1, rows1, cols), arr2(_arr2, rows2, cols);
    array1i idx(_idx, rows1, 1);
    
    FOR(ii, 0, arr1.rows()) {
        FOR(jj, 0, arr2.rows()) {
            dlon = arr1(ii, 0) - arr2(jj, 0);
            dlat = arr1(ii, 1) - arr2(jj, 1);
            
            if ((dlon * dlon + dlat * dlat) < max_sep) {
                idx(ii) = 1;
                nfound++;
                break;
            }
        }
    }    
}

void test(double * _array, int n, int m)
{
    array2d arr(_array, n, m);
    
    FOR(ii, 0, arr.rows()) {
        FOR(jj, 0, arr.cols())
            cout << arr(ii,jj) << " ";
        cout << endl;
    }
} // test

#endif
