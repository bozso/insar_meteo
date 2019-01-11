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

#include "array.hpp"

// Structure for storing fitted polynom coefficients
struct fit_poly {
    double mean_t, start_t, stop_t, *mean_coords;
    View<double> const& coeffs;
    size_t is_centered, deg;
    fit_poly(double mean_t, double start_t, double stop_t, double *mean_coords,
             View<double> const& coeffs, size_t is_centered, size_t deg):
        mean_t(mean_t), start_t(start_t), stop_t(stop_t),
        mean_coords(mean_coords), coeffs(coeffs), is_centered(is_centered),
        deg(deg)
        {}
    ~fit_poly() {};
};

#endif
