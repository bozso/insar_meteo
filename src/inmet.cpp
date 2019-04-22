#include <exception>

#include "aux.hpp"
#include "math.hpp"

using std::exception;
using std::cout;
using std::cerr;

using aux::inarray;
using aux::outarray;
using aux::inpoly;
using aux::idx;
using aux::end;
//using aux::print;


extern "C" {

int test(inarray _a)
{
    try {
        auto a = _a->array<double, 1>();
        //printf("Pointers: %p %p\n", _a->data, asd);

        cout << aux::type_info(_a->type).name << " " << _a->shape[0] << end;

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
        
        for (idx ii = 0; ii < _a->shape[0]; ++ii) {
            //cout << a(ii) << " ";
            sum += a(ii);
        }
        
        printf("\nSum: %15.10g\n", sum);

        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}


int eval_poly_c(inpoly poly, inarray x, inarray y)
{
    try {
        aux::eval_poly(poly, x, y);
        return 0;
    }
    catch(exception& e) {
        cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }

}

// extern "C"
}
