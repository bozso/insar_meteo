#include <exception>
#include <iostream>

#include "array.hpp"
#include "math.hpp"

using std::exception;
using std::cout;
using std::cerr;


using aux::arr_in;
using aux::arr_out;



extern "C" {


int aaa(arr_in a, arr_out b)
{
    try {
        auto const va = a.const_view<float>(1);
        auto vb = b.view<double>(1);
        
        for (int ii = 0; ii < a.shape[0]; ++ii) {
            printf("%f\t%f\n", va(ii), vb(ii));
        }

        printf("\n");
        
        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}


int eval_poly(math::poly_in poly, arr_in x, arr_out y)
{
    try {
        printf("%ld %p\n", poly.coeffs.ndim, poly.coeffs.data);
        math::eval_poly(poly, x, y);
        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }

}


// extern "C"
}
