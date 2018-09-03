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

#include "utils.h"

/***********
 * Structs *
 ***********/

// structure for storing fitted orbit polynom coefficients
typedef struct orbit_fit_t {
    double mean_t, start_t, stop_t;
    double *mean_coords, *coeffs;
    uint is_centered, deg;
} orbit_fit;

// cartesian coordinate
typedef struct cart_t {
    double x, y, z;
} cart;

void ell_cart (cdouble, cdouble, cdouble, double *, double *, double *);

void cart_ell(cdouble, cdouble, cdouble, double *, double *, double *);

void calc_azi_inc(const orbit_fit *, cdouble, cdouble, cdouble, cdouble,
                  cdouble, cdouble, cuint, double *, double *);

#endif
