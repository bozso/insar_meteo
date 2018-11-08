#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "utils.hh"

#define array_type(ar_struct) &((ar_struct).pyobj)


enum dtype {
    dt_double = NPY_DOUBLE,
    dt_bool = NPY_BOOL
};


struct nparray {
    int typenum;
    size_t ndim, psize, *shape, *strides;
    PyArrayObject *npobj;
    PyObject *pyobj;
    bool decref;

    nparray():
            typenum(0), ndim(0), psize(0), shape(NULL), strides(NULL),
            npobj(NULL), pyobj(NULL), decref(false) {}

    
    nparray(size_t const ndim, int const typenum):
            typenum(typenum), ndim(ndim), psize(ndim * 2 * sizeof(size_t)),
            shape(NULL), strides(NULL), npobj(NULL), pyobj(NULL), decref(false)
            {};
    
    ~nparray() {
        if (decref)
            Py_CLEAR(npobj);
    }

    bool from_data(Pool& pool, npy_intp* dims, void *data);
    bool import(Pool& pool, PyObject* obj = NULL);
    bool empty(Pool& pool, npy_intp* dims, int const fortran = 0);
    bool zeros(Pool& pool, npy_intp* dims, int const fortran = 0);
    PyArrayObject* ret();
    void * data() const;
    
    bool check_rows(npy_intp const rows) const;
    bool check_cols(npy_intp const cols) const;
    bool is_f_cont() const;
};



#endif
