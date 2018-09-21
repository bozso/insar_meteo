#define ARMA_INCLUDE_MEAT
#include "armadillo.hh"

using namespace arma;

#define eT double

//template
//bool
//internal_approx_equal_abs_diff<eT>(const eT& x, const eT& y, const typename get_pod_type<eT>::result tol);

template class Mat<eT>;

#undef eT
