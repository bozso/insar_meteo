//#include "pyvector.hh"

#define __INMET_IMPL

#include "carray.hh"
#include "nparray.hh"

template struct nparray<npy_double, 2>;
template struct nparray<npy_double, 1>;
template struct nparray<npy_bool, 1>;

template struct array<npy_double, 2>;
template struct array<npy_double, 1>;
template struct array<npy_bool, 1>;

//template struct vector<double>;

template struct carray<bool>;

#undef __INMET_IMPL
