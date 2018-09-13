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

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000.0

#define WA  6378137.0
#define WB  6356752.3142

// (WA^2 - WB^2) / WA^2
#define E2  6.694380e-03

/************************
 * DEGREES, RADIANS, PI *
 ************************/

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01
#define distance(x, y, z) sqrt((y)*(y)+(x)*(x)+(z)*(z))

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

struct _errdsc {
    int code;
    char * message;
} errdsc[] = {
    { err_succes, "No error encountered."},
    { err_io, "IO error encountered!" },
    { err_alloc, "Allocation error encountered!" },
    { err_num, "Numerical error encountered!" },
    { err_arg, "Command line argument error encountered!" }
};

// WGS-84 surface coordinates
struct llh { double lon, lat, h; } llh; 

// Orbit records
struct orbit_rec { double t, x, y, z; };

int fit_orbit(int argc, char **argv);
int azi_inc(int argc, char **argv);
int eval_orbit(int argc, char **argv);
int test_matrix1(void);
int test_matrix2(void);

#endif
