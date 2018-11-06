#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#define array_type(ar_struct) &((ar_struct).pyobj)


#ifdef __INMET_IMPL
template<class T> struct dtype { static const int typenum; };

template<>
const int dtype<npy_double>::typenum = NPY_DOUBLE;

template<>
const int dtype<npy_bool>::typenum = NPY_BOOL;

#endif

enum dtype {
    dt_double = NPY_DOUBLE,
    dt_bool = NPY_BOOL
};


struct nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    PyObject *pyobj;
    bool decref;

    nparray():
            typenum(0), ndim(0), shape(NULL), strides(NULL),
            data(NULL), npobj(NULL), pyobj(NULL), decref(true) {}

    
    nparray(size_t const ndim, int const typenum):
            typenum(typenum), ndim(ndim), shape(NULL), strides(NULL),
            data(NULL), npobj(NULL), pyobj(NULL), decref(true) {}
    
    ~nparray() {
        if (shape != NULL)
            PyMem_Del(shape);

        if (strides != NULL)
            PyMem_Del(strides);
        
        if (decref)
            Py_CLEAR(npobj);
    }
};


bool const from_data(nparray& arr, npy_intp* dims, void *data);
bool const import(nparray& arr, PyObject* _obj = NULL);
bool const empty(nparray& arr, npy_intp* dims, int const fortran = 0);
bool const zeros(nparray& arr, npy_intp* dims, int const fortran = 0);
size_t const rows(nparray const& arr);
size_t const cols(nparray const& arr);
PyArrayObject* ret(nparray& arr);

bool const check_rows(nparray const& arr, npy_intp const rows);
bool const check_cols(nparray const& arr, npy_intp const cols);
bool const is_f_cont(nparray const& arr);

#endif