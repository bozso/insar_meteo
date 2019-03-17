#include <iostream>
#include <exception>

#include "numpy.hpp"

using std::cout;
using std::cerr;

namespace np = numpy;

extern "C" {


    int test(np::array_ptr a)
    {
        try
        {
            cout << "Asd\n";
        }
        catch(std::exception& e)
        {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
