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
#include "nparray.h"
#include "common.h"

extern_begin

/***********
 * Structs *
 ***********/

// cartesian coordinate
typedef struct cart {
    double x, y, z;
} cart;

typedef double const cdouble;

void ell_cart (cdouble lon, cdouble lat, cdouble h,
               double *x, double *y, double *z);

void cart_ell(cdouble x, cdouble y, cdouble z,
              double *lon, double *lat, double *h);

void calc_azi_inc(fit_poly const *orb, nparray const _coords,
                  nparray __azi_inc, size_t const max_iter,
                  uint const is_lonlat);

extern_end

#endif // SATORBIT_H
