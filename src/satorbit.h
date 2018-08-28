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

// structure for storing fitted orbit polynom coefficients
typedef struct orbit_fit_t {
    double mean_t;
    double * mean_coords;
    double start_t, stop_t;
    double * coeffs;
    uint is_centered, deg;
} orbit_fit;

// cartesian coordinate
typedef struct cart_t { double x, y, z; } cart;

void ell_cart (cdouble lon, cdouble lat, cdouble h,
               double *x, double *y, double *z);

void cart_ell (cdouble x, cdouble y, cdouble z,
               double *lon, double *lat, double *h);

extern void calc_azi_inc(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, cdouble lon, cdouble lat,
                         const uint max_iter, double * azi, double * inc);

#endif
