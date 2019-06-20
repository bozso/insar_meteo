#include "math.hpp"

namespace math {

using aux::arr_in;
using aux::arr_out;
using aux::idx;


template<class T>
static void eval_poly_tpl(int nfit, arr_in coeffs, arr_in ncoeffs,
                          arr_in X, arr_out Y)
{
    auto x = X.const_view<T>(1);
    auto y = Y.view<T>(2);
    auto vcoeffs = coeffs.const_view<T>(1);
    auto vncoeffs = ncoeffs.const_view<idx>(1);

    for (idx ii = 0; ii < X.shape[0]; ++ii) {
        idx start = 0, stop = vncoeffs(0);
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
            
            T sum = vcoeffs(start);
            
            for (idx jj = start + 1; jj < stop - 1; ++jj) {
                sum = vcoeffs(jj) +  sum * _x;
            }
            
            y(ii, nn) = vcoeffs(stop) + sum * _x;
            
            start += vncoeffs(nn);
            stop += vncoeffs(nn + 1);
        }
    }
}


void eval_poly(int nfit, arr_in coeffs, arr_in ncoeffs,
               arr_in x, arr_out y)
{
    auto const is_cpx = coeffs.get_type().is_complex;
    
    if (is_cpx) {
        eval_poly_tpl<double>(nfit, coeffs, ncoeffs, x, y);
    } else {
        eval_poly_tpl<aux::cpxd>(nfit, coeffs, ncoeffs, x, y);
    }

    
    /*
    auto const& type = coeffs.get_type().id;
    
    switcher2(eval_poly_tpl, type, poly, x, y);
    */
}

// namespace end
}