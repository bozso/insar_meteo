#include <stdarg.h>

#include "capi_functions.hh"

template<typename T> struct dtype { static const int typenum; };

template<>
const int dtype<npy_double>::typenum = NPY_DOUBLE;

template<>
const int dtype<npy_bool>::typenum = NPY_BOOL;


void pyexc(PyObject *exception, const char *format, ...)
{
    va_list vl;
    va_start(vl, format);
    PyErr_FormatV(exception, format, vl);
    va_end(vl);
}


template<typename T, unsigned int ndim>
array<T, ndim>::array(T * _data, ...)
{
    va_list vl;
    unsigned int shape_sum = 0;
    
    va_start(vl, _data);
    
    for(unsigned int ii = 0; ii < ndim; ++ii)
        shape[ii] = static_cast<unsigned int>(va_arg(vl, int));
    
    va_end(vl);
    
    for(unsigned int ii = 0; ii < ndim; ++ii) {
        shape_sum = 1;
        
        for(unsigned int jj = ii + 1; jj < ndim; ++jj)
             shape_sum *= shape[jj];
        
        strides[ii] = shape_sum;
    }
    data = _data;
}


template<typename T, unsigned int ndim>
bool array<T, ndim>::setup_array(PyArrayObject *__array)
{
    int _ndim = static_cast<unsigned int>(PyArray_NDIM(__array));
    
    if (ndim != _ndim) {
        pyexc(PyExc_TypeError, "numpy array expected to be %u "
                               "dimensional but we got %u dimensional "
                               "array!", ndim, _ndim);
        return true;
        
    }
    
    npy_intp * _shape = PyArray_DIMS(__array);

    for(unsigned int ii = 0; ii < ndim; ++ii)
        shape[ii] = static_cast<unsigned int>(_shape[ii]);

    int elemsize = int(PyArray_ITEMSIZE(__array));
    
    npy_intp * _strides = PyArray_STRIDES(__array);
    
    for(unsigned int ii = 0; ii < ndim; ++ii)
        strides[ii] = static_cast<unsigned int>(double(_strides[ii])
                                                    / elemsize);
    
    data = (T*) PyArray_DATA(__array);
    _array = __array;
    
    return false;
}


template<typename T, unsigned int ndim>
bool array<T, ndim>::import(PyObject *__obj)
{
    PyObject * tmp_obj = NULL;
    PyArrayObject * tmp_array = NULL;
    
    if (_obj == NULL)
        tmp_obj = _obj;
    else
        _obj = tmp_obj = __obj;
    
    if ((tmp_array =
         (PyArrayObject*) PyArray_FROM_OTF(tmp_obj, dtype<T>::typenum,
                                           NPY_ARRAY_IN_ARRAY)) == NULL) {
        pyexc(PyExc_TypeError, "Failed to convert numpy array!");
        return true;
    }
    
    return setup_array(tmp_array);
}


template<typename T, unsigned int ndim>
bool array<T, ndim>::empty(npy_intp *dims, int fortran)
{
    PyArrayObject * tmp_array = NULL;
    
    if ((tmp_array = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        pyexc(PyExc_TypeError, "Failed to create empty numpy array!");
        return 1;
    }
    
    return setup_array(tmp_array);
}


template class array<npy_double, 2>;
template class array<npy_double, 1>;

template class array<npy_bool, 1>;
