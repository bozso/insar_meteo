#pragma once

namespace aux {

struct PolyFit {
    idx nfit;
    inarray coeffs, ncoeffs;
};

typedef cptr<PolyFit const> inpoly;

void eval_poly(inpoly poly, inarray x, inarray y);

}