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

/************************
 * Structs and typedefs *
 ************************/

typedef unsigned int uint;
typedef const double cdouble;

struct orbit_fit {
    double mean_t;
    double * mean_coords;
    double start_t, stop_t;
    double * coeffs;
    uint is_centered, deg;
};

struct cart {double x, y, z; };

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA  6378137.0
#define WB  6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2  6.694380e-03

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

void calc_pos(const orbit_fit& orb, double time, cart& pos);

double dot_product(const orbit_fit& orb, cdouble X, cdouble Y,
                   cdouble Z, double time);

void closest_appr(const orbit_fit& orb, cdouble X, cdouble Y,
                  cdouble Z, const uint max_iter, cart& sat_pos);

inline void calc_azi_inc(const orbit_fit& orb, cdouble X, cdouble Y,
                         cdouble Z, cdouble lon, cdouble lat,
                         const uint max_iter, double& azi, double& inc);


template<class T>
inline T deg2rad(const T deg)
{
    return deg * 1.745329e-02;
}

template<class T>
inline T rad2deg(const T rad)
{
    return rad * 5.729578e+01;
}

template<class T>
inline T norm(const T x, const T y, const T z)
{
    // vector norm
    return sqrt(x * x + y * y + z * z);
}

template<class T>
static void cart_ell (const T x, const T y, const T z,
                      T& lon, T& lat, T& h)
{
    // from cartesian to ellipsoidal coordinates
    
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
}
// end of cart_ell

template<class T>
static void ell_cart (const T lon, const T lat, const T h, T& x, T& y, T& z)
{
    // from ellipsoidal to cartesian coordinates
    
    T n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));;

    x = (              n + h) * cos(lat) * cos(lon);
    y = (              n + h) * cos(lat) * sin(lon);
    z = ( (1.0 - E2) * n + h) * sin(lat);
} // end of ell_cart

#endif
