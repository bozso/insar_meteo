#include "aux.hpp"
#include "math.hpp"

namespace aux {


template<class T>
void eval_poly_tpl(inpoly poly, inarray X, outarray Y)
{
    auto const& nfit = poly->nfit;
    auto x = X->view<T, 1>();
    auto y = Y->view<T, 2>();
    auto coeffs = poly->coeffs->array<T, 1>();
    auto ncoeffs = poly->ncoeffs->array<idx, 1>();
    
    for (idx ii = 0; ii < X->shape[0]; ++ii) {
        idx start = 0, stop = ncoeffs(0);
        auto const& _x = x(ii);
        
        for (idx nn = 0; nn < nfit; ++nn) {
            /*
               nfit = 3
               ncoeffs = (3, 4, 2)
               0. 1. 2. | 3. 4. 5. 6. | 7. 8.
               
               nn = 0, start = 0, stop = 3
               jj = 0,1,2

               nn = 1
               start = 3, stop = 7
               jj = 3, 4, 5, 6
               
               nn = 2
               start = 7, stop = 9
               jj = 7, 8
             */
            
            T sum = coeffs(start);
            
            for (idx jj = start + 1; jj < stop - 1; ++jj) {
                sum = coeffs(jj) +  sum * _x;
            }
            
            y(ii, nn) = coeffs(stop) + sum * _x;
            
            start += ncoeffs(nn);
            stop += ncoeffs(nn + 1);
        }
    }
}


void eval_poly(inpoly poly, inarray x, outarray y)
{
    auto const& type = poly->coeffs->get_type().id;
    
    switcher(eval_poly_tpl, type, poly, x, y);
}

// namespace aux
}