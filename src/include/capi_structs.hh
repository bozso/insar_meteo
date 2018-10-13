#ifndef CAPI_STRUCTS_HH
#define CAPI_STRUCTS_HH

#include "Python.h"
#include "numpy/arrayobject.h"

void pyexc(PyObject *exception, const char *format, ...);

template<typename T, unsigned int ndim>
struct array {
    unsigned int shape[ndim], strides[ndim];
    PyArrayObject *_array;
    PyObject *_obj;
    T * data;
    
    array() data(NULL), _array(NULL), _obj(NULL) {};
    array(T * _data, ...);
    
    bool setup_array(PyArrayObject *_array);
    bool import(PyObject *__obj = NULL);
    bool empty(npy_intp *dims, int fortran = 0);
    
    ~array();
    
    PyArrayObject * get_array() const;
    PyObject * get_obj() const;
    T* get_data() const;
    
    const unsigned int get_shape(unsigned int ii) const;
    const unsigned int rows() const;
    const unsigned int cols() const;
    
    T& operator()(unsigned int ii);
    T& operator()(unsigned int ii, unsigned int jj);
    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk);
    T& operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                  unsigned int ll);
};


template<typename T>
struct vector {
    T* data;
    size_t cnt;
    size_t cap;
    
    vector() data(NULL), cnt(0), cap(0) {};
    bool init(size_t buf_cap);
    void init(T* buf, size_t buf_cap);
    
    ~vector();
    
    push(T& elem);
    
};


template <typename T>
struct array {
    T* data;
    size_t size;
    
    array();
    
    bool init(const size_t& init_size);
    bool init(const int& init_size, const T init_value);
    bool init(const array& original);
    
    ~array();
    
    T& operator[](size_t index);
    T const& operator[] const (size_t index);
    array& operator= (const array& copy);
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
