#include "capi_structs.hh"
//#include "pyvector.hh"
#include "array.hh"
#include "capi_array.hh"

template struct nparray<npy_double, 2>;
template struct nparray<npy_double, 1>;
template struct nparray<npy_bool, 1>;

//template struct vector<double>;

template struct array<bool>;

