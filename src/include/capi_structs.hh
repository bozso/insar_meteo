#ifndef CAPI_STRUCTS_HH
#define CAPI_STRUCTS_HH

#include <stddef.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "capi_macros.hh"

void pyexc(PyObject *exception, const char *format, ...);

template<class T, size_t ndim>
struct nparray {
    unsigned int shape[ndim], strides[ndim];
    PyArrayObject *_array;
    PyObject *_obj;
    T * data;
    
    nparray(): _array(NULL), _obj(NULL), data(NULL) {};
    nparray(T *_data, ...);

    bool const import(PyObject *__obj = NULL);
    bool const empty(npy_intp *dims, int const fortran = 0);
    
    ~nparray();
    
    PyArrayObject * get_array() const;
    PyObject * get_obj() const;
    T* get_data() const;
    
    size_t const get_shape(size_t ii) const;
    size_t const rows() const;
    size_t const cols() const;
    
    T& operator()(size_t const ii);
    T& operator()(size_t const ii, size_t const jj);
    T& operator()(size_t const ii, size_t const jj, size_t const kk);
    T& operator()(size_t const ii, size_t const jj, size_t const kk,
                  size_t const ll);

    T const operator()(size_t const ii) const;
    T const operator()(size_t const ii, size_t const jj) const;
    T const operator()(size_t const ii, size_t const jj, size_t const kk) const;
    T const operator()(size_t const ii, size_t const jj, size_t const kk,
                       size_t const ll) const;
};


template<class T>
struct vector {
    T* data;
    size_t cnt;
    size_t cap;
    
    vector(): data(NULL), cnt(0), cap(0) {};
    bool const init(size_t const buf_cap);
    void init(T* buf, size_t const buf_cap);
    
    bool const push(T const& elem);

    bool const add(T* vals, size_t const n);
    T* addn(size_t const n, bool const init0);
    
    ~vector();
};


template <class T>
struct array {
    T* data;
    size_t size;
    
    array(): data(NULL), size(0) {};

    bool const init(size_t const init_size);
    bool const init(size_t const init_size, T const init_value);
    bool const init(array<T> const& original);

    ~array();
    
    T& operator[](size_t const index);
    T const operator[](size_t const index) const;
    array& operator= (array const & copy);
};


#if 0

template<typename T>
struct arraynd {
    unsigned int *strides;
    npy_intp *shape;
    PyArrayObject *_array;
    T * data;
    
    arraynd()
    {
        strides = NULL;
        shape = NULL;
        data = NULL;
        _array = NULL;
    };
    
    ~arraynd()
    {
        if (strides != NULL) {
            PyMem_Del(strides);
            strides = NULL;
        }
    }

    const PyArrayObject * get_array()
    {
        return _array;
    }
    
    const unsigned int get_shape(unsigned int ii)
    {
        return static_cast<unsigned int>(shape[ii]);
    }

    const unsigned int rows()
    {
        return static_cast<unsigned int>(shape[0]);
    }
    
    const unsigned int cols()
    {
        return static_cast<unsigned int>(shape[1]);
    }
    
    T* get_data()
    {
        return data;
    }
    
    T& operator()(unsigned int ii)
    {
        return data[ii * strides[0]];
    }

    T& operator()(unsigned int ii, unsigned int jj)
    {
        return data[ii * strides[0] + jj * strides[1]];
    }
    
    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk)
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }

    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                  unsigned int ll)
    {
        return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[3]];
    }
};

template<typename T>
static int import(arraynd<T>& arr, PyArrayObject * _array = NULL) {

    PyArrayObject * tmp_array = NULL;
    
    if (_array == NULL)
        tmp_array = arr._array;
    else
        arr._array = tmp_array = _array;
    
    unsigned int ndim = static_cast<unsigned int>(PyArray_NDIM(tmp_array));
    arr.shape = PyArray_DIMS(tmp_array);
    
    int elemsize = int(PyArray_ITEMSIZE(tmp_array));
    
    if ((arr.strides = PyMem_New(unsigned int, ndim)) == NULL) {
        pyexcs(PyExc_MemoryError, "Failed to allocate memory for array "
                                  "strides!");
        return 1;
    }
    
    npy_intp * _strides = PyArray_STRIDES(tmp_array);
    
    for(unsigned int ii = 0; ii < ndim; ++ii)
        arr.strides[ii] = static_cast<unsigned int>(double(_strides[ii])
                                                    / elemsize);
    
    arr.data = static_cast<T*>(PyArray_DATA(tmp_array));
    
    return 0;
}

#endif

#endif
