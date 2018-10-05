#include <stdarg.h>

#include "capi_functions.hh"

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
array<T, ndim>::bool setup_array(PyArrayObject *_array)
{
    int _ndim = static_cast<unsigned int>(PyArray_NDIM(_array));
    
    if (ndim != _ndim) {
        pyexc(PyExc_TypeError, "numpy array expected to be %u "
                               "dimensional but we got %u dimensional "
                               "array!", ndim, _ndim);
        return true;
        
    }
    
    npy_intp * _shape = PyArray_DIMS(_array);

    for(unsigned int ii = 0; ii < ndim; ++ii)
        arr.shape[ii] = static_cast<unsigned int>(_shape[ii]);

    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    npy_intp * _strides = PyArray_STRIDES(_array);
    
    for(unsigned int ii = 0; ii < ndim; ++ii)
        arr.strides[ii] = static_cast<unsigned int>(double(_strides[ii])
                                                    / elemsize);
    
    arr.data = (T*) PyArray_DATA(_array);
    arr._array = _array;
    
    return false;
}


template<typename T, unsigned int ndim>
array<T, ndim>::bool import(PyObject *__obj)
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
        pyexcs(PyExc_TypeError, "Failed to convert numpy array!");
        return true;
    }
    
    return setup_array(tmp_array);
}


template<typename T, unsigned int ndim>
array<T, ndim>::bool empty(npy_intp *dims, int fortran)
{
    PyArrayObject * tmp_array = NULL;
    
    if ((tmp_array = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        pyexcs(PyExc_TypeError, "Failed to create empty numpy array!");
        return 1;
    }
    
    return setup_array(tmp_array);
}


template<> array<npy_double, 2>;
template<> array<npy_double, 1>;

template<> array<npy_bool, 1>;
