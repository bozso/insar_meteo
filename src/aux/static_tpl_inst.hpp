#ifndef STL_INST_HPP
#define STL_INST_HPP

#include <vector>
#include <memory>

//#define m_spec_xarray(T) \
//xt::xarray_container<XTENSOR_DEFAULT_DATA_CONTAINER(T, XTENSOR_DEFAULT_ALLOCATOR(T)), XTENSOR_DEFAULT_LAYOUT, XTENSOR_DEFAULT_SHAPE_CONTAINER(T, (XTENSOR_DEFAULT_ALLOCATOR(T)), (std::allocator<typename std::vector<T, A>::size_type>>))>

extern template class std::vector<double>;
//extern template struct m_spec_xarray(double);

#endif
