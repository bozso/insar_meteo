#ifndef ARRAY_HH
#define ARRAY_HH

#include <stddef.h>


template<class T, size_t ndim>
struct array {
    size_t shape[ndim], strides[ndim];
    T * data;
    
    array(): data(NULL) {};
    array(T *_data, ...);

    T& operator()(size_t const ii);
    T& operator()(size_t const ii, size_t const jj);
    T& operator()(size_t const ii, size_t const jj, size_t const kk);
    T& operator()(size_t const ii, size_t const jj, size_t const kk,
                  size_t const ll);

    T const operator()(size_t const ii) const;
    T const operator()(size_t const ii, size_t const jj) const;
    T const operator()(size_t const ii, size_t const jj, size_t const kk) const;
    T const operator()(size_t const ii, size_t const jj, size_t const kk,
                       size_t const ll) const;
};

#ifdef __INMET_IMPL

template<typename T, size_t ndim>
array<T, ndim>::array(T * _data, ...)
{
    va_list vl;
    size_t shape_sum = 0;
    
    va_start(vl, _data);
    
    for(size_t ii = 0; ii < ndim; ++ii)
        shape[ii] = size_t(va_arg(vl, int));
    
    va_end(vl);
    
    for(size_t ii = 0; ii < ndim; ++ii) {
        shape_sum = 1;
        
        for(size_t jj = ii + 1; jj < ndim; ++jj)
             shape_sum *= shape[jj];
        
        strides[ii] = shape_sum;
    }
    data = _data;
}


template<typename T, size_t ndim>
T& array<T, ndim>::operator()(size_t const ii) {
    return data[ii * strides[0]];
}


template<typename T, size_t ndim>
T& array<T, ndim>::operator()(size_t const ii, size_t const jj) {
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, size_t ndim>
T& array<T, ndim>::operator()(size_t const ii, size_t const jj, size_t const kk) {
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, size_t ndim>
T& array<T, ndim>::operator()(size_t const ii, size_t const jj, size_t const kk,
                                size_t const ll) {
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

template<typename T, size_t ndim>
T const array<T, ndim>::operator()(size_t const ii) const {
    return data[ii * strides[0]];
}


template<typename T, size_t ndim>
T const array<T, ndim>::operator()(size_t const ii, size_t const jj) const {
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, size_t ndim>
T const array<T, ndim>::operator()(size_t const ii, size_t const jj,
                                     size_t const kk) const {
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, size_t ndim>
T const array<T, ndim>::operator()(size_t const ii, size_t const jj,
                                     size_t const kk, size_t ll) const {
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

#endif

#endif
