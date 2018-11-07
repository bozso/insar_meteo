#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#define array_type(ar_struct) &((ar_struct).pyobj)


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
            npobj(NULL), pyobj(NULL), decref(false) {}

    
    nparray(size_t const ndim, int const typenum):
            typenum(typenum), ndim(ndim), shape(NULL), strides(NULL),
            npobj(NULL), pyobj(NULL), decref(false) {}
    
    ~nparray() {
        if (shape != NULL)
            PyMem_Del(shape);

        if (strides != NULL)
            PyMem_Del(strides);
        
        if (decref)
            Py_CLEAR(npobj);
    }

    bool const from_data(npy_intp* dims, void *data);
    bool const import(PyObject* obj = NULL);
    bool const empty(npy_intp* dims, int const fortran = 0);
    bool const zeros(npy_intp* dims, int const fortran = 0);
    PyArrayObject* ret();
    void * data() const;
    
    bool const check_rows(npy_intp const rows) const;
    bool const check_cols(npy_intp const cols) const;
    bool const is_f_cont() const;
};



#endif
