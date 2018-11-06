#ifndef ARRAY_HH
#define ARRAY_HH

#include <stddef.h>

#include "nparray.hh"

template<class T>
struct view {
    T * data;
    size_t ndim, *shape, *strides;

    view(): data(NULL) {};
    view(T *data, size_t ndim, size_t *shape, size_t *strides):
         data(data), ndim(ndim), shape(shape), strides(strides) {};

    view(nparray const& arr): data((T*) PyArray_DATA(arr.npobj)), ndim(arr.ndim),
                              shape(arr.shape), strides(arr.strides) {};
    
    ~view() {};

    #if 0
    #ifndef __INMET_IMPL
    view(T *_data, ...);
    #else
    view(T * _data, ...) {
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
    #endif
    #endif
    
    T& operator()(size_t const ii) {
        return data[ii * strides[0]];
    }
    
    T& operator()(size_t const ii, size_t const jj) {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    
    T& operator()(size_t const ii, size_t const jj, size_t const kk) {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }
    
    T& operator()(size_t const ii, size_t const jj, size_t const kk,
                                    size_t const ll) {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }
    
    T const operator()(size_t const ii) const {
        return data[ii * strides[0]];
    }
    
    T const operator()(size_t const ii, size_t const jj) const {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    T const operator()(size_t const ii, size_t const jj, size_t const kk) const {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }
    
    T const operator()(size_t const ii, size_t const jj, size_t const kk, size_t ll) const {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }

};

#endif
