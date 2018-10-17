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


template<typename T, size_t ndim>
nparray<T, ndim>::nparray(T * _data, ...)
{
    va_list vl;
    size_t shape_sum = 0;
    
    va_start(vl, _data);
    
    for(size_t ii = 0; ii < ndim; ++ii)
        shape[ii] = size_t(va_arg(vl, int));
    
    va_end(vl);
    
    for(size_t ii = 0; ii < ndim; ++ii) {
        shape_sum = 1;
        
        for(size_t jj = ii + 1; jj < ndim; ++jj)
             shape_sum *= shape[jj];
        
        strides[ii] = shape_sum;
    }
    data = _data;
}


template<typename T, size_t ndim>
nparray<T, ndim>::~nparray() {
    Py_CLEAR(_array);
}


template<typename T, size_t ndim>
static bool const setup_array(nparray<T, ndim> *arr, PyArrayObject *_array)
{
    int _ndim = size_t(PyArray_NDIM(_array));
    
    if (ndim != _ndim) {
        pyexc(PyExc_TypeError, "numpy nparray expected to be %u "
                               "dimensional but we got %u dimensional "
                               "nparray!", ndim, _ndim);
        return true;
        
    }
    
    npy_intp * shape = PyArray_DIMS(_array);

    for(size_t ii = 0; ii < ndim; ++ii)
        arr->shape[ii] = size_t(shape[ii]);

    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = 0; ii < ndim; ++ii)
        arr->strides[ii] = size_t(double(strides[ii]) / elemsize);
    
    arr->data = (T*) PyArray_DATA(_array);
    arr->_array = _array;
    
    return false;
}


template<typename T, size_t ndim>
bool const nparray<T, ndim>::import(PyObject *__obj)
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
        pyexc(PyExc_TypeError, "Failed to convert numpy nparray!");
        return true;
    }
    
    return setup_array(this, tmp_array);
}


template<typename T, size_t ndim>
bool const nparray<T, ndim>::empty(npy_intp *dims, int const fortran)
{
    PyArrayObject * tmp_array = NULL;
    
    if ((tmp_array = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        pyexc(PyExc_TypeError, "Failed to create empty numpy nparray!");
        return true;
    }
    
    return setup_array(this, tmp_array);
}


template<typename T, size_t ndim>
PyArrayObject* nparray<T, ndim>::get_array() const {
    return _array;
}


template<typename T, size_t ndim>
PyObject* nparray<T, ndim>::get_obj() const {
    return _obj;
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::get_shape(size_t const ii) const {
    return shape[ii];
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::rows() const {
    return shape[0];
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::cols() const {
    return shape[1];
}


template<typename T, size_t ndim>
T* nparray<T, ndim>::get_data() const {
    return data;
}


template<typename T, size_t ndim>
T& nparray<T, ndim>::operator()(size_t const ii) {
    return data[ii * strides[0]];
}


template<typename T, size_t ndim>
T& nparray<T, ndim>::operator()(size_t const ii, size_t const jj) {
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, size_t ndim>
T& nparray<T, ndim>::operator()(size_t const ii, size_t const jj, size_t const kk) {
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, size_t ndim>
T& nparray<T, ndim>::operator()(size_t const ii, size_t const jj, size_t const kk,
                                size_t const ll) {
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

template<typename T, size_t ndim>
T const nparray<T, ndim>::operator()(size_t const ii) const {
    return data[ii * strides[0]];
}


template<typename T, size_t ndim>
T const nparray<T, ndim>::operator()(size_t const ii, size_t const jj) const {
    return data[ii * strides[0] + jj * strides[1]];
}


template<typename T, size_t ndim>
T const nparray<T, ndim>::operator()(size_t const ii, size_t const jj,
                                     size_t const kk) const {
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}


template<typename T, size_t ndim>
T const nparray<T, ndim>::operator()(size_t const ii, size_t const jj,
                                     size_t const kk, size_t ll) const {
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                + ll * strides[3]];
}

#endif
