#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

using namespace xt;

int main()
{
    xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};
    
    xarray<double> arr2
      {5.0, 6.0, 7.0};
    
    xarray<double> res = view(arr1, 1) + arr2;
    
    std::cout << res << std::endl;
}
