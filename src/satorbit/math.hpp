#ifndef __MATH_HPP
#define __MATH_HPP

#include "array.hpp"

namespace math {

struct PolyFit {
    aux::idx nfit;
    aux::arr_in coeffs, ncoeffs;
};


using poly_in = PolyFit const&;
using poly_out = PolyFit&;

void eval_poly(poly_in poly, aux::arr_in x, aux::arr_out y);

}

// guard
#endif