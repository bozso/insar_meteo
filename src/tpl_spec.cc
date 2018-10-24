//#include "pyvector.hh"

#define __INMET_IMPL

#include "nparray.hh"
#include "array.hh"

template struct nparray<npy_double>;
template struct nparray<npy_bool>;

//template struct vector<double>;

template struct array<bool>;

#undef __INMET_IMPL
