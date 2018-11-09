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
            npobj(NULL), pyobj(NULL), decref(false)
            {};
    
    ~nparray() {
        PyMem_Del(strides);
        strides = shape = NULL;
        
        if (decref)
            Py_CLEAR(npobj);
    }


    bool from_data(int const typenum, size_t const ndim, npy_intp* dims,
                   void *data);
    
    bool import(int const typenum, size_t const ndim = 0, PyObject* obj = NULL);
    
    bool empty(int const typenum, size_t const ndim, npy_intp* dims,
               int const fortran = 0);
    
    bool zeros(int const typenum, size_t const ndim, npy_intp* dims,
               int const fortran = 0);
    
    PyArrayObject* ret();
    void * data() const;
    
    bool check_rows(npy_intp const rows) const;
    bool check_cols(npy_intp const cols) const;
    bool is_f_cont() const;
};



#endif
