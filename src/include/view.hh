#ifndef VIEW_HH
#define VIEW_HH

#include "nparray.hh"

template<class T>
struct view {
    T *data;
    size_t ndim, *shape, *strides;

    view(): data(NULL), ndim(0), shape(NULL), strides(NULL) {};
    //view(T *data, ssize_t const ndim, ssize_t const* shape, ssize_t const* strides):
         //data(data), ndim(ndim), shape(shape), strides(strides)
    //{ setup_view(strides, ndim, sizeof(T)) };
    
    view(nparray const& arr): data((T*) PyArray_DATA(arr.npobj)), ndim(arr.ndim),
                              shape(arr.shape), strides(arr.strides) {};
    
    
    
    ~view() {};


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
