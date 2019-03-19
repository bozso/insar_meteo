#include <exception>

#include "aux.hpp"

using std::cout;
using std::cerr;


using aux::array_ptr;
using aux::Array;
using aux::DArray;
using aux::View;
using aux::idx;
using aux::print;

extern "C" {
    int test(array_ptr const _a)
    {
        try {
            DArray const a(_a);

            
            for (idx ii = 0; ii < a.shape(0); ++ii)
                printf("%d ", a.get<int>(ii));
            
            print("\n");
            
            auto aa = aux::type_info(aux::dtype::Int);
            print("%\n", aa.is_complex);
        }
        catch(std::exception& e) {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
