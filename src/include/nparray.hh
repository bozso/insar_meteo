#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "array.hh"

#define array_type(ar_struct) &((ar_struct).pyobj)
#define ret(ar_struct) (ar_struct).npobj


template<class T, size_t ndim>
struct nparray {
    array<T, ndim> arr;
    PyArrayObject *npobj;
    PyObject *pyobj;
    bool decref;
    
    nparray() {
        npobj = NULL;
        pyobj = NULL;
        arr.data = NULL;
        decref = false;
    }
    
    bool const from_data(npy_intp *dims, void *data);
    bool const import(PyObject *_obj = NULL);
    bool const empty(npy_intp *dims, int const fortran = 0, bool const decref = false);
    bool const zeros(npy_intp *dims, int const fortran = 0, bool const decref = false);
    
    ~nparray() {
        if (decref)
            Py_CLEAR(npobj);
    }
    
    PyObject * get_obj() const;
    T* get_data() const;
    
    bool const is_f_cont() const;
    
    size_t const get_shape(size_t ii) const;
    size_t const rows() const;
    size_t const cols() const;
};


#ifdef __INMET_IMPL

template<typename T> struct dtype { static const int typenum; };

template<>
const int dtype<npy_double>::typenum = NPY_DOUBLE;

template<>
const int dtype<npy_bool>::typenum = NPY_BOOL;


template<typename T, size_t ndim>
static bool const setup_array(array<T, ndim>& arr, PyArrayObject *_array, bool const checkdim = false)
{
    int _ndim = size_t(PyArray_NDIM(_array));
    
    if (checkdim and ndim != _ndim) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    ndim, _ndim);
        return true;
        
    }
    
    npy_intp * shape = PyArray_DIMS(_array);

    for(size_t ii = 0; ii < ndim; ++ii)
        arr.shape[ii] = size_t(shape[ii]);

    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = 0; ii < ndim; ++ii)
        arr.strides[ii] = size_t(double(strides[ii]) / elemsize);
    
    arr.data = (T*) PyArray_DATA(_array);
    
    return false;
}

template<typename T, size_t ndim>
bool const nparray<T, ndim>::from_data(npy_intp *dims, void *data)
{
    if ((npobj = (PyArrayObject*) PyArray_SimpleNewFromData(ndim, dims,
                      dtype<T>::typenum, data)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj);
}


template<typename T, size_t ndim>
bool const nparray<T, ndim>::import(PyObject *_obj)
{
    if (_obj != NULL)
        pyobj = _obj;
    
    if ((npobj =
         (PyArrayObject*) PyArray_FROM_OTF(pyobj, dtype<T>::typenum,
                                           NPY_ARRAY_IN_ARRAY)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to convert numpy nparray!");
        return true;
    }
    
    decref = true;
    return setup_array(arr, npobj, true);
}


template<typename T, size_t ndim>
bool const nparray<T, ndim>::empty(npy_intp *dims, int const fortran, bool const decref)
{
    if ((npobj = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj);
}


template<typename T, size_t ndim>
bool const nparray<T, ndim>::zeros(npy_intp *dims, int const fortran, bool const decref)
{
    if ((npobj = (PyArrayObject*) PyArray_ZEROS(ndim, dims,
                      dtype<T>::typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj);
}



template<typename T, size_t ndim>
PyObject* nparray<T, ndim>::get_obj() const {
    return pyobj;
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::get_shape(size_t const ii) const {
    return arr.shape[ii];
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::rows() const {
    return arr.shape[0];
}


template<typename T, size_t ndim>
size_t const nparray<T, ndim>::cols() const {
    return arr.shape[1];
}


template<typename T, size_t ndim>
T* nparray<T, ndim>::get_data() const {
    return arr.data;
}

template<typename T, size_t ndim>
bool const nparray<T, ndim>::is_f_cont() const {
    return PyArray_IS_F_CONTIGUOUS(npobj);
}

#endif

#endif
