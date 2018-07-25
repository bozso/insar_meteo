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

#ifndef MAIN_FUN_H
#define MAIN_FUN_H

#include <iostream>

using namespace std;

#define min_arg 2

#define aux_checkarg(num, doc)\
({\
    if (argc != ((num) + min_arg)) {\
        cerr << "\n Required number of arguments is "<< (num) <<\
                ", current number of arguments: " << argc - min_arg << "!\n";\
        cerr << (doc) << endl;\
        return -1;\
    }\
})

/*******************************
 * WGS-84 ellipsoid parameters *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000.0

#define WA  6378137.0
#define WB  6356752.3142

// (WA^2 - WB^2) / WA^2
#define E2  6.694380e-03

/************************
 * degrees, radians, pi *
 ************************/

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

/***************
 * Error codes *
 ***************/

enum err_code {
    err_succes = 0,
    err_io = -1,
    err_alloc = -2,
    err_num = -3,
    err_arg = -4
};

// Idx -- column major order
#define Idx(ii, jj, nrows) (ii) + (jj) * (nrows)

typedef unsigned int uint;
typedef const double cdouble;

// Cartesian coordinates
struct cart { double x, y, z; }; 

// WGS-84 surface coordinates
struct llh { double lon, lat, h; }; 

// Orbit records
struct orbit { double t, x, y, z; };

// Fitted orbit polynoms structure
struct orbit_fit {
    uint deg, centered;
    double t_min, t_max, t_mean;
    double *coeffs, coords_mean[3];
};

int fit_orbit(int argc, char **argv);
//void azi_inc(int argc, char **argv);
//void eval_orbit(int argc, char **argv);

#endif
