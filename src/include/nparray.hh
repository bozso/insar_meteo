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

enum newtype {
    empty,
    zeros
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
    
    bool import(int const typenum, size_t const ndim = 0, PyObject* obj = NULL);
    bool newarr(int const typenum, void *data, size_t num, ...);
    bool newarr(int const typenum, newtype const newt, int const fortran, size_t num, ...);
    
    PyArrayObject* ret();
    void * data() const;
    
    bool check_rows(size_t const rows) const;
    bool check_cols(size_t const cols) const;
    bool is_f_cont() const;
};

#if 0

struct shapei {
    npy_intp *dims;
    size_t ndim;
    
    shapei(): dims(NULL) {};
    
    bool init(int num, ...)
    {
        ndim = num;
        
        if (dims != NULL)
            delete[] ndim;

        if ((dims = new npy_intp[num]) == NULL)
            return true;
        
        va_list vl;

        va_start(vl, num);

        for(size_t ii = 0; ii < num; ++ii)
            dims[ii] = va_arg(vl, npy_intp);
        
        va_end(vl);
    }
    
    ~shapei() {
        delete[] dims;
    };
}
#endif

#endif
