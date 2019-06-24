#include <exception>
#include <iostream>

#include "array.hpp"
// #include "math.hpp"


using std::exception;
using std::cout;
using std::cerr;


extern "C" {

int aaa(aux::Array& a)
{
    try {
        auto const va = a.view<int64_t>(1);
        
        for (int ii = 0; ii < va.shape(0); ++ii) {
            printf("%ld\n", va(ii));
        }

        printf("\n");
        
        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}

/*
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
*/

// extern "C"
}
