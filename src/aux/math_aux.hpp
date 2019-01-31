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

#include <vector>
#include <complex>

using std::vector;
using std::complex;


#include "array.hpp"

// Structure for storing fitted polynom coefficients

class Number {
    dtype type;
    union {
        bool b;
        long il;
        int  ii;
        ssize_t is;
    
        int8_t i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
    
        uint8_t  ui8;
        uint16_t ui16;
        uint32_t ui32;
        uint64_t ui64;
    
        float f32;
        double f64;
    
        complex<float> c64;
        complex<double> c128;
    } val;
    
    Number() = default;

    template<typename T>
    Number(T& num)
    {
        if std::is_same<T, bool> {
            this->val.b = num;
            this->type = dt_bool
        } else if std::is_same<T, long> {
            
        }
         
        
        
    }
}        

struct MinMax {
    Number min, max;
    MinMax() = min(0), max(0);
    ~MinMax() = default;
};

struct Scale {
    Number min, scale;
    Scale() : min(0), scale(0);
    
    Scale(MinMax const& m) : min(m.min), scale(m.max - m.min) {};
    ~Scale() = default;
};

class PolyFit {
    Array coeffs;
    int deg;
    bool scaled;
    union {Scale<T>; void} xs;
    union {vector<Scale<T>>; void} ys;
    
    PolyFit(Array const& x, Array const& y, int deg, bool scaled);
    ~PolyFit;
}
    
    
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
