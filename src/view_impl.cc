#include "utils.hh"

void setup_view(ssize_t *strides, size_t const ndim, size_t const itemsize)
{
    FORZ(ii, ndim)
        strides[ii] = ssize_t(double(strides[ii]) / itemsize);
}
