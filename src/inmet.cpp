#include <iostream>
#include <exception>

#include "numpy.hpp"

using std::cout;
using std::cerr;


extern "C" {
    int test(array_ptr a)
    {
        try
        {
            auto mtx = from_numpy<float>(a);
            
            for (int ii = 0; ii < mtx.rows(); ++ii)
            {
                printf("%f ", mtx(0, ii));
            }
            cout << "\n";
        }
        catch(std::exception& e)
        {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
