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
            auto a = _a->array<double, 1>();
            
            //idx const ii = 0;
            //aux::memptr data = a.data + ii * a.strides[0];
            
            //cout << a.convert(data) << end;
            //cout << a(0) << " " << a(1) << " " << a(2) << end;
            //cout << *reinterpret_cast<double*>(data) << end;
            //cout << static_cast<double>(*reinterpret_cast<double*>(data)) << end;
            
            for(idx ii = 0; ii < 15; ++ii) {
                cout << a(ii) << " ";
            }
            
            //cout << "\nLast: " << a(a.array.shape[0] - 1) << end;

            double sum = 0.0;
            //
            for (idx ii = 0; ii < _a->shape[0]; ++ii) {
                sum += a(ii);
            }
                    
            cout << "Sum: " << sum << end;
            return 0;
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";

            return 1;
        }
        return 0;
    }
}
