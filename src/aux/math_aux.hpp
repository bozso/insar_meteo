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

struct Number {
    ArrayMeta::dtype type;
    union {
        bool   b;
        long   il;
        int    ii;
        size_t is;
    
        int8_t  i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
    
        uint8_t  ui8;
        uint16_t ui16;
        uint32_t ui32;
        uint64_t ui64;
    
        float  fl32;
        double fl64;
    
        complex<float>  c64;
        complex<double> c128;
    };
    
    Number() = delete;
    Number(bool n)            : b(n) {};
    Number(int  n)            : ii(n) {};
    Number(long n)            : il(n) {};
    Number(size_t n)          : is(n) {};
    Number(int8_t n)          : i8(n) {};
    Number(int16_t n)         : i16(n) {};
    Number(int32_t n)         : i32(n) {};
    Number(int64_t n)         : i64(n) {};

    Number(uint8_t n)         : ui8(n) {};
    Number(uint16_t n)        : ui16(n) {};
    Number(uint32_t n)        : ui32(n) {};
    Number(uint64_t n)        : ui64(n) {};
    
    Number(float n)           : fl32(n) {};
    Number(double n)          : fl64(n) {};
    
    Number(complex<float> n)  : c64(n) {};
    Number(complex<double> n) : c128(n) {};
    ~Number() = default;
};        

struct MinMax {
    Number min, max;
    MinMax() : min(0), max(0) {};
    ~MinMax() = default;
};

struct Scale {
    Number min, scale;
    Scale() : min(0), scale(0);
    
    Scale(MinMax const& m) : min(m.min), scale(m.max - m.min) {};
    ~Scale() = default;
};

/*
class PolyFit {
    Array coeffs;
    int deg;
    bool scaled;
    union {Scale; void} xs;
    union {uarr<Scale>; void} ys;
    
    PolyFit(Array const& x, Array const& y, int deg, bool scaled);
    ~PolyFit() = default;
}
*/    
    
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
