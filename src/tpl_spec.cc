//#include "pyvector.hh"

#define __INMET_IMPL

#include "nparray.hh"
#include "view.hh"
#include "array.hh"

template struct nparray<npy_double, 2>;
template struct nparray<npy_double, 1>;
template struct nparray<npy_bool, 1>;

//template struct vector<double>;

template struct array<bool>;

#undef __INMET_IMPL
