#include "capi_structs.hh"
#include "pyvector.hh"
#include "array.hh"
#include "capi_array.hh"

#define spec_arr(type)\
template struct array<type>;\
template bool init(array<type>& arr, const size_t init_size);\
template bool init(array<type>& arr, const int init_size, const T init_value);\
template bool init(array<type>& arr, const array& original);\


template struct nparray<npy_double, 2>;
template struct nparray<npy_double, 1>;
template struct nparray<npy_bool, 1>;

template struct vector<double>;

spec_arr(bool)

#undef __INMET_INC_IMPL
