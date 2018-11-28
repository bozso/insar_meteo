#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

size_t calc_size(size_t num_array, size_t numdim);

enum dtype {
    dt_double = NPY_DOUBLE,
    dt_bool = NPY_BOOL
};

enum newtype {
    empty,
    zeros
};


struct nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    bool _decref;
    
    nparray():
            typenum(0), ndim(0), shape(NULL), strides(NULL),
            npobj(NULL), _decref(false) {}

    
    nparray(size_t const ndim, int const typenum):
            typenum(typenum), ndim(ndim), shape(NULL), strides(NULL),
            npobj(NULL), _decref(false)
            {};
    
    nparray(int const typenum, size_t const ndim, PyObject* obj);
    nparray(int const typenum, void *data, size_t num, ...);
    nparray(int const typenum, newtype const newt, char const layout, size_t num, ...);
    
    ~nparray();
    
    PyArrayObject* ret();
    
    template<class T>
    T* data() const { return (T*) PyArray_DATA(npobj); };
    
    bool check_rows(size_t const rows) const;
    bool check_cols(size_t const cols) const;
    bool is_f_cont() const;
};

#endif
