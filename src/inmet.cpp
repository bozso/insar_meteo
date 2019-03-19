#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::array_ptr;
using aux::idx;
//using aux::print;

extern "C" {
    int test(array_ptr const _a)
    {
        try {
            auto a = _a->view<double>();
            
            
            
            for (idx ii = 0; ii < a.shape(0); ++ii)
                printf("%f ", a(ii));
            
            
            printf("\n");
            
            auto aa = aux::type_info(aux::dtype::Int);
            printf("%d\n", aa.is_complex);
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
