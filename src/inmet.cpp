#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>

using namespace xt;
using namespace std;

#define size 25000

template<class T>
xtensor<T, 2>& read(string infile)
{
    xtensor<T, 2> * mtx;
    
    mtx = new xtensor<T, 2>(size, size);
    
    return mtx;
}

int main()
{
    auto mtx = read<double>("");
    
    for(uint ii = 0; ii < size; ++ii)
        for(uint jj = 0; jj < size; ++jj)
            mtx(ii, jj) = ii + jj;
    
    cout << view(mtx, 0, range(0, 10)) << endl;
    
    return 0;
}
