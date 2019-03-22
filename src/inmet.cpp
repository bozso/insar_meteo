#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::array_ptr;
using aux::idx;
using aux::end;
//using aux::print;


extern "C" {
    int test(array_ptr const _a)
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
