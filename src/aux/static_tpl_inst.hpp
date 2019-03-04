#ifndef STL_INST_HPP
#define STL_INST_HPP

#include <vector>
#include <memory>

#include <xtensor/xarray.h>

extern template class std::vector<double>;
extern template struct xt::xarray<double>;

#endif
