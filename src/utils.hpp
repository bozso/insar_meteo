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

#ifndef UTILS_H
#define UTILS_H

#include <cstdio>

typedef unsigned int uint;
typedef const unsigned int cuint;
typedef const double cdouble;

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA 6378137.0
#define WB 6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2 6.694380e-03


#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); ++(ii))
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/*************
 * IO macros *
 *************/

#define _log fprintf(stderr, "File: %s line: %d", __FILE__, __LINE__)

#endif
