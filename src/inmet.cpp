#include <exception>
#include <iostream>

//#include "aux.hpp"
#include "array.hpp"
//#include "math.hpp"

using std::exception;
using std::cout;
using std::cerr;


using numpy::arr_in;
using numpy::arr_out;


extern "C" {

int aaa(arr_in a)
{
    auto const va = a.view<int64_t>(1);
    // printf("ndim: %ld name: %s\n", a.ndim, a.get_type().name);
    
    for (int ii = 0; ii < a.shape[0]; ++ii) {
        printf("%ld ", va(ii));
    }
    printf("\n");
    
    return 0;
}


/*
int test(inarray a)
{
    try {
        auto va = a.array<double>(1);
        //printf("Pointers: %p %p\n", _a->data, asd);

        //cout << aux::type_info(_a->type).name << " " << _a->shape[0] << end;

        //idx const ii = 0;
        //aux::memptr data = a.data + ii * a.strides[0];
        
        //cout << a.convert(data) << end;
        //cout << a(0) << " " << a(1) << " " << a(2) << end;
        //cout << *reinterpret_cast<double*>(data) << end;
        //cout << static_cast<double>(*reinterpret_cast<double*>(data)) << end;
        
        //printf("\t%15s\t|\t%15s\t\n", "Carray", "Pointer");
        //
        //for(idx ii = 0; ii < 15; ++ii) {
            //printf("\t%15.5g\t|\t%15.5g\t\n", a(ii), asd[ii]);
        //}
        //
        //return 0;
        
        //cout << "\nLast: " << a(a.array.shape[0] - 1) << end;
        
        double sum = 0.0;
        
        for (idx ii = 0; ii < a.shape[0]; ++ii) {
            //cout << a(ii) << " ";
            sum += va(ii);
        }
        
        printf("\nSum: %15.10g\n", sum);

        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}

int eval_poly_c(int nfit, inarray coeffs, inarray ncoeffs,
                inarray x, outarray y)
{
    try {
        aux::eval_poly(nfit, coeffs, ncoeffs, x, y);
        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }

}

int aaa(int& ii)
{
    cout << "C++: Number before: " << ii << end;
    ii++;
    cout << "C++: Number after: " << ii << end;

    return 0;
}
*/


// extern "C"
}
