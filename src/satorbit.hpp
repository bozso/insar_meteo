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

#ifndef SATORBIT_H
#define SATORBIT_H

#include <cmath>

/************************
 * Structs and typedefs *
 ************************/

// structure for storing fitted orbit polynom coefficients
struct orbit_fit {
    double mean_t;
    double * mean_coords;
    double start_t, stop_t;
    double * coeffs;
    size_t is_centered, deg;
};

template<typename T>
static inline T norm(const T x, const T y, const T z)
{
    return sqrt(x * x + y * y + z * z);
}

// cartesian coordinate
struct cart { double x, y, z; };

inline void im_ell_cart (cdouble lon, cdouble lat, cdouble h,
                         double *x, double *y, double *z)
{
    // from ellipsoidal to cartesian coordinates
    
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    *x = (              n + h) * cos(lat) * cos(lon);
    *y = (              n + h) * cos(lat) * sin(lon);
    *z = ( (1.0 - E2) * n + h) * sin(lat);

} // end of ell_cart


inline void im_cart_ell (cdouble x, cdouble y, cdouble z,
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


extern void im_ell_cart (cdouble lon, cdouble lat, cdouble h,
                         double *x, double *y, double *z);

extern void im_cart_ell (cdouble x, cdouble y, cdouble z,
                         double *lon, double *lat, double *h);

extern void im_calc_azi_inc(const orbit_fit * orb, cdouble X, cdouble Y,
                            cdouble Z, cdouble lon, cdouble lat,
                            size_t max_iter, double * azi,
                            double * inc);

void calc_azi_inc(double start_t, double stop_t, double mean_t,
                  double * mean_coords, double * coeffs, int is_centered,
                  int deg, int max_iter, int is_lonlat, double * coords,
                  int n_coords, double * azi_inc);

extern "C"{

void test(double * array, int n, int m);

}

#endif
