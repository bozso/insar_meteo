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


#ifndef MATH_AUX_HH
#define MATH_AUX_HH

#include "common.h"
#include "view.h"

extern_begin

/* structure for storing fitted polynom coefficients */
typedef struct fit_poly {
    double mean_t, start_t, stop_t, *mean_coords;
    view_double *coeffs;
    size_t is_centered, deg;
} fit_poly;

extern_end

#endif
