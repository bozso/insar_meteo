#include <iostream>
#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::array_ptr;

extern "C" {
    int test(array_ptr a)
    {
        try {
            
            
            auto a = aux::TypeInfo<aux::cpx64>::make_info();
            cout << a.is_complex << "\n";
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
