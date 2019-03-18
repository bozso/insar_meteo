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
            auto info = aux::type_info(aux::dtype::Bool);
            cout << info.id << "\n";
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
