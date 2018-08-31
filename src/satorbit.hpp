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
#include "utils.hpp"

namespace inmet {

/***********
 * Structs *
 ***********/

// structure for storing fitted orbit polynom coefficients
struct orbit_fit {
    double mean_t, start_t, stop_t;
    double *mean_coords, *coeffs;
    uint is_centered, deg;
    
    orbit_fit() {};
    
    orbit_fit(double _mean_t, double _start_t, double _stop_t,
              double * _mean_coords, double * _coeffs, uint _is_centered,
              uint _deg)
    {
        mean_t = _mean_t;
        start_t = _start_t;
        stop_t = _stop_t;
        
        mean_coords = _mean_coords;
        coeffs = _coeffs;
        
        is_centered = _is_centered;
        deg = _deg;
    }
};

// cartesian coordinate
struct cart {
    double x, y, z;
    
    cart() {};
    
    cart(double _x, double _y, double _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
};

template<typename T>
static inline T norm(const T x, const T y, const T z)
{
    return sqrt(x * x + y * y + z * z);
}

// from ellipsoidal to cartesian coordinates
template<typename T>
inline void ell_cart (const T lon, const T lat, const T h,
                      T& x, T& y, T& z)
{
    T n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    x = (              n + h) * cos(lat) * cos(lon);
    y = (              n + h) * cos(lat) * sin(lon);
    z = ( (1.0 - E2) * n + h) * sin(lat);

} // ell_cart

// from cartesian to ellipsoidal coordinates
template<typename T>
inline void im_cart_ell (const T x, const T y, const T z,
                         T& lon, T& lat, T& h)
{
    T n, p, o, so, co;

    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    lat = o;
    
    o = atan(y/x); if(x < 0.0) o += M_PI;
    lon = o;
    h = p / co - n;
} // cart_ell


extern "C"{

void azi_inc(double start_t, double stop_t, double mean_t,
             double * mean_coords, double * coeffs, int is_centered,
             int deg, int max_iter, int is_lonlat, double * _coords,
             int n_coords, double * _azi_inc);

void test(double * array, int n, int m);

} // extern "C"

} //namespace inmet

#endif
