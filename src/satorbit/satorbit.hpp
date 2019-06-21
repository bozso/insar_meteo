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

#ifndef __SATORBIT_H
#define __SATORBIT_H


#include "array.hpp"

/***********
 * Structs *
 ***********/

/* cartesian coordinate */
struct cart {
    double x = 0.0, y = 0.0, z = 0.0;
    
    cart() = default;
    ~cart() = default;
};


struct Ellipsoid {
    double a = 0.0, b = 0.0, e2 = 0.0;
    
    Ellipsoid() = default;
    ~Ellipsoid() = default;
};


extern "C" {
    void set_ellipsoid(aux::cptr<Ellipsoid const> new_ellipsoid);
    void print_ellipsoid();
}


// TODO: move constants into separate header

/********************
 * Useful constants *
 ********************/
namespace consts {
    
    constexpr double pi = 3.14159265358979;
    constexpr double pi_per_4 = 3.14159265358979 / 4.0;
    
    constexpr double deg2rad = 1.745329e-02;
    constexpr double rad2deg = 5.729578e+01;
}


// guard
#endif 
