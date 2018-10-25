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


template<class T>
struct nparray {
    npy_intp *shape, *strides;
    T * data;
    PyArrayObject *npobj;
    PyObject *pyobj;
    bool decref;
    
    nparray(): shape(NULL), strides(NULL), data(NULL), npobj(NULL),
               pyobj(NULL) {}
    
    ~nparray() {
        if (strides != NULL)
            PyMem_Del(strides);
        
        if (decref)
            Py_CLEAR(npobj);
    }
    

    #ifndef __INMET_IMPL
    bool const from_data(size_t const ndim, npy_intp const* dims, void *data);
    bool const import(size_t const ndim, PyObject *_obj = NULL);
    bool const empty(size_t const ndim, npy_intp const* dims,
                     int const fortran = 0);
    bool const zeros(size_t const ndim, npy_intp const* dims,
                     int const fortran = 0);

    PyObject * get_obj() const;
    T* get_data() const;
    
    bool const is_f_cont() const;
    
    size_t const get_shape(size_t ii) const;
    size_t const rows() const;
    size_t const cols() const;
    PyArrayObject* ret() const;

    bool const check_rows(size_t const rows) const;
    bool const check_cols(size_t const cols) const;


    #else

    bool const from_data(size_t const ndim, npy_intp const* dims, void *data)
    {
        if ((npobj = (PyArrayObject*) PyArray_SimpleNewFromData(ndim, dims,
                          dtype<T>::typenum, data)) == NULL) {
            PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
            return true;
        }
        
        return setup_array(this, npobj, 0);
    }
    
    
    bool const import(size_t const ndim, PyObject *_obj = NULL)
    {
        if (_obj != NULL)
            pyobj = _obj;
        
        if ((npobj =
             (PyArrayObject*) PyArray_FROM_OTF(pyobj, dtype<T>::typenum,
                                               NPY_ARRAY_IN_ARRAY)) == NULL) {
            PyErr_Format(PyExc_TypeError, "Failed to convert numpy nparray!");
            return true;
        }
        
        return setup_array(this, npobj, ndim);
    }
    
    
    bool const empty(size_t const ndim, npy_intp const* dims,
                     int const fortran = 0)
    {
        if ((npobj = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                          dtype<T>::typenum, fortran)) == NULL) {
            PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
            return true;
        }
        
        return setup_array(this, npobj, 0);
    }
    

    bool const zeros(size_t const ndim, npy_intp const* dims,
                     int const fortran = 0)
    {
        if ((npobj = (PyArrayObject*) PyArray_ZEROS(ndim, dims,
                          dtype<T>::typenum, fortran)) == NULL) {
            PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
            return true;
        }
        
        return setup_array(this, npobj, 0);
    }
    
    PyObject* get_obj() const {
        return pyobj;
    }
    
    
    size_t const rows() const {
        return size_t(shape[0]);
    }

    size_t const cols() const {
        return size_t(shape[1]);
    }
    
    PyArrayObject* ret() const
    {
        decref = false;
        return npobj;
        
    }
    
    
    
    bool const check_rows(npy_intp const rows) const
    {
        if (shape[0] != rows) {
            PyErr_Format(PyExc_TypeError, "Expected array to have rows %u but got "
                         "array with rows %u.", rows, shape[0]);
            return true;
        }
        return false;
    }
    
    
    bool const check_cols(npy_intp const cols) const
    {
        if (shape[1] != cols) {
            PyErr_Format(PyExc_TypeError, "Expected array to have cols %u but got "
                         "array with cols %u.", cols, shape[1]);
            return true;
        }
        return false;
    }
    
    
    T* get_data() const {
        return data;
    }
    
    bool const is_f_cont() const {
        return PyArray_IS_F_CONTIGUOUS(npobj);
    }

    #endif
    

    T& operator()(size_t const ii) {
        return data[ii * strides[0]];
    }
    
    T& operator()(size_t const ii, size_t const jj) {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    T& operator()(size_t const ii, size_t const jj, size_t const kk) {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }
    
    T& operator()(size_t const ii, size_t const jj, size_t const kk, size_t const ll) {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }
    
    T const operator()(size_t const ii) const {
        return data[ii * strides[0]];
    }
    
    T const operator()(size_t const ii, size_t const jj) const {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    T const operator()(size_t const ii, size_t const jj, size_t const kk) const {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }
    
    T const operator()(size_t const ii, size_t const jj, size_t const kk, size_t ll) const {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }
};



#if 0

template<typename T, size_t ndim>
view<T> nparray<T, ndim>::get_view()
{
    view<T> retv(data, ndim, shape, strides);
    return retv;
}


template<typename T, size_t ndim>
view<T> const nparray<T, ndim>::get_view() const
{
    const view<T> retv(data, ndim, shape, strides);
    return retv;
}

#endif


template<class T>
static bool const setup_array(nparray<T> *arr, PyArrayObject *_array,
                              size_t const edim = 0)
{
    int _ndim = size_t(PyArray_NDIM(_array));
    
    if (edim and edim != _ndim) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return true;
        
    }
    
    arr->shape = PyArray_DIMS(_array);

    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    if ((arr->strides = PyMem_New(npy_intp, _ndim)) == NULL) {
        
        return true;
    }
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->strides[ii] = size_t(double(strides[ii]) / elemsize);
    
    arr->data = (T*) PyArray_DATA(_array);
    
    return false;
}

#endif
