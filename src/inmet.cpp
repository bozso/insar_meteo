#include <iostream>
#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::array_ptr;
using aux::View;
using aux::idx;

extern "C" {
    int test(array_ptr _a)
    {
        try {
            View<float> a(_a);
            
            
            for (idx ii = 0; ii < a.shape(0); ++ii)
                cout << a(ii) << " ";
            
            cout << "\n";
            
            //auto a = aux::type_info(aux::dtype::Int);
            //cout << a.is_complex << "\n";
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
