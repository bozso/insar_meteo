#ifndef CAPI_STRUCTS_HH
#define CAPI_STRUCTS_HH

#include "Python.h"
#include "numpy/arrayobject.h"

#include "capi_macros.hh"

void pyexc(PyObject *exception, const char *format, ...);

template<class T, unsigned int ndim>
struct nparray {
    unsigned int shape[ndim], strides[ndim];
    PyArrayObject *_array;
    PyObject *_obj;
    T * data;
    
    nparray(): _array(NULL), _obj(NULL), data(NULL) {};
    nparray(T *_data, ...);
    
    ~nparray();
    
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

    const T operator()(unsigned int ii) const;
    const T operator()(unsigned int ii, unsigned int jj) const;
    const T operator()(unsigned int ii, unsigned int jj, unsigned int kk) const;
    const T operator()(unsigned int ii, unsigned int jj, unsigned int kk,
                  unsigned int ll) const;
};

template<typename T, unsigned int ndim>
bool import(nparray<T, ndim>& arr, PyObject *__obj = NULL);

template<typename T, unsigned int ndim>
bool empty(nparray<T, ndim>& arr, npy_intp *dims, int fortran = 0);

template<typename T, unsigned int ndim>
PyArrayObject* get_array(const nparray<T, ndim>& arr);

template<typename T, unsigned int ndim>
PyObject* get_obj(const nparray<T, ndim>& arr);

template<typename T, unsigned int ndim>
const unsigned int get_shape(const nparray<T, ndim>& arr, unsigned int ii);

template<typename T, unsigned int ndim>
const unsigned int rows(const nparray<T, ndim>& arr);

template<typename T, unsigned int ndim>
const unsigned int cols(const nparray<T, ndim>& arr);

template<typename T, unsigned int ndim>
T* get_data(const nparray<T, ndim>& arr);



static const size_t DG__DYNARR_SIZE_T_MSB = ((size_t)1) << (sizeof(size_t)*8 - 1);
static const size_t DG__DYNARR_SIZE_T_ALL_BUT_MSB = (((size_t)1) << (sizeof(size_t)*8 - 1))-1;

template<class T>
struct vector {
    T* data;
    size_t cnt;
    size_t cap;
    
    vector(): data(NULL), cnt(0), cap(0) {};
    bool init(size_t buf_cap);
    void init(T* buf, size_t buf_cap);
    
    DG_DYNARR_DEF static bool grow(size_t min_needed);
    DG_DYNARR_INLINE bool maybegrowadd(const size_t num_add);
    DG_DYNARR_INLINE bool maybegrow(const size_t min_needed);
    bool push(const T& elem);

    DG_DYNARR_INLINE bool add(const size_t n, const bool init0);
    bool add(T* vals, const size_t n);
    T* addn(size_t n, bool init0);
    
    DG_DYNARR_INLINE bool _insert(const size_t idx, const size_t n, const bool init0);
    
    static void checkidx(size_t ii);
    
    
    ~vector();

    
};


template <class T>
struct array {
    T* data;
    size_t size;
    
    array(): data(NULL), size(0) {};
    ~array();
    
    T& operator[](size_t index);
    const T operator[](size_t index) const;
    array& operator= (const array& copy);
};


template <class T>
bool init(array<T>& arr, const size_t init_size);

template <class T>
bool init(array<T>& arr, const int init_size, const T init_value);

template <class T>
bool init(array<T>& arr, const array<T>& original);


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
