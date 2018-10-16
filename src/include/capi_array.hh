#ifndef CAPI_nparray_HH
#define CAPI_nparray_HH

#include <stdarg.h>

#include "capi_structs.hh"

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
nparray<T, ndim>::nparray(T * _data, ...)
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
bool nparray<T, ndim>::setup_nparray(PynparrayObject *__nparray)
{
    int _ndim = static_cast<unsigned int>(Pynparray_NDIM(__nparray));
    
    if (ndim != _ndim) {
        pyexc(PyExc_TypeError, "numpy nparray expected to be %u "
                               "dimensional but we got %u dimensional "
                               "nparray!", ndim, _ndim);
        return true;
        
    }
    
    npy_intp * _shape = Pynparray_DIMS(__nparray);

    for(unsigned int ii = 0; ii < ndim; ++ii)
        shape[ii] = static_cast<unsigned int>(_shape[ii]);

    int elemsize = int(Pynparray_ITEMSIZE(__nparray));
    
    npy_intp * _strides = Pynparray_STRIDES(__nparray);
    
    for(unsigned int ii = 0; ii < ndim; ++ii)
        strides[ii] = static_cast<unsigned int>(double(_strides[ii])
                                                    / elemsize);
    
    data = (T*) Pynparray_DATA(__nparray);
    _nparray = __nparray;
    
    return false;
}


template<typename T, unsigned int ndim>
bool nparray<T, ndim>::import(PyObject *__obj)
{
    PyObject * tmp_obj = NULL;
    PynparrayObject * tmp_nparray = NULL;
    
    if (_obj == NULL)
        tmp_obj = _obj;
    else
        _obj = tmp_obj = __obj;
    
    if ((tmp_nparray =
         (PynparrayObject*) Pynparray_FROM_OTF(tmp_obj, dtype<T>::typenum,
                                           NPY_nparray_IN_nparray)) == NULL) {
        pyexc(PyExc_TypeError, "Failed to convert numpy nparray!");
        return true;
    }
    
    return setup_nparray(tmp_nparray);
}


template<typename T, unsigned int ndim>
bool nparray<T, ndim>::empty(npy_intp *dims, int fortran)
{
    PynparrayObject * tmp_nparray = NULL;
    
    if ((tmp_nparray = (PynparrayObject*) Pynparray_EMPTY(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        pyexc(PyExc_TypeError, "Failed to create empty numpy nparray!");
        return 1;
    }
    
    return setup_nparray(tmp_nparray);
}


template<typename T, unsigned int ndim>
nparray<T, ndim>::~nparray()
{
    Py_XDECREF(_nparray);
}


template<typename T, unsigned int ndim>
PynparrayObject* nparray<T, ndim>::get_nparray() const
{
    return _nparray;
}


template<typename T, unsigned int ndim>
PyObject* nparray<T, ndim>::get_obj() const
{
    return _obj;
}


template<typename T, unsigned int ndim>
const unsigned int nparray<T, ndim>::get_shape(unsigned int ii) const
{
    return shape[ii];
}


template<typename T, unsigned int ndim>
const unsigned int nparray<T, ndim>::rows() const
{
    return shape[0];
}


template<typename T, unsigned int ndim>
const unsigned int nparray<T, ndim>::cols() const
{
    return shape[1];
}


template<typename T, unsigned int ndim>
T* nparray<T, ndim>::get_data() const
{
    return data;
}


template<typename T, unsigned int ndim>
T& nparray<T, ndim>::operator()(unsigned int ii)
{
    return data[ii * strides[0]];
}


template<typename T, unsigned int ndim>
T& nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj)
{
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, unsigned int ndim>
T& nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj, unsigned int kk)
{
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, unsigned int ndim>
T& nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                              unsigned int ll)
{
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

template<typename T, unsigned int ndim>
const T nparray<T, ndim>::operator()(unsigned int ii) const
{
    return data[ii * strides[0]];
}


template<typename T, unsigned int ndim>
const T nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj) const
{
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, unsigned int ndim>
const T nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj, unsigned int kk) const
{
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, unsigned int ndim>
const T nparray<T, ndim>::operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                              unsigned int ll) const
{
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

#endif
