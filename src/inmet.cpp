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
        
        //auto itf = a.interface();
        //auto const& type = aux::detail::get_type(itf.typekind, itf.itemsize);
        // printf("%c\n", a.descr().kind);
        // printf("%c %d\n", itf.typekind, itf.itemsize);
        //printf("Name: %s\n", type.name);
        /*
        auto const va = a.const_view<float>(1);
        auto vb = b.view<double>(1);
        
        for (int ii = 0; ii < a.shape[0]; ++ii) {
            printf("%f\t%f\n", va(ii), vb(ii));
        }

        printf("\n");
        */
        
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
