#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::inarray;
using aux::outarray;
using aux::idx;
using aux::end;
//using aux::print;


extern "C" {
    int test(inarray _a)
    {
        try {
            cout << _a->check_ndim(2) << end;
            cout << _a->check_shape(3, aux::row) << end;
            
            auto a = _a->array<double>();

            for (idx ii = 0; ii < _a->shape[0]; ++ii) {
                printf("%lf ", a(ii));
            }
            
            printf("\n");
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
