#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>

#include "numpy/ndarrayobject.h"

enum newtype {
    empty,
    zeros
};


struct nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    bool _decref;
    
    nparray():
            typenum(0), ndim(0), shape(NULL), strides(NULL),
            npobj(NULL), _decref(false) {}

    
    nparray(size_t const ndim, int const typenum):
            typenum(typenum), ndim(ndim), shape(NULL), strides(NULL),
            npobj(NULL), _decref(false)
            {};
    
    
    nparray(int const typenum, size_t const ndim, PyObject* obj);
    nparray(size_t const ndim, PyObject* obj);
    nparray(int const typenum, void *data, size_t num, ...);
    nparray(int const typenum, newtype const newt, char const layout, size_t num, ...);
    nparray(int const typenum, size_t const ndim, PyArrayObject *obj);
    ~nparray();
    
    PyArrayObject* ret();
    
    template<class T>
    T* data() const { return (T*) PyArray_DATA(npobj); };
    
    bool is_f_cont() const;
    bool check_rows(size_t const rows) const;
    bool check_cols(size_t const cols) const;
    bool is_zero_dim() const;
    bool check_scalar() const;
    bool is_python_number() const;
    bool is_python_scalar() const;
    bool is_not_swapped() const;
    bool is_byte_swapped() const;
    bool can_cast_to(int const totypenum);
    nparray cast_to_type(int const typenum);
};


enum dtype {
    np_bool = NPY_BOOL,

    np_byte = NPY_BYTE,
    np_ubyte = NPY_UBYTE,
    np_short = NPY_SHORT,
    np_ushort = NPY_USHORT,
    np_int = NPY_INT,
    np_uint = NPY_UINT,
    np_long = NPY_LONG,
    np_ulong = NPY_ULONG,
    np_longlong = NPY_LONGLONG,
    np_ulonglong = NPY_ULONGLONG,

    np_double = NPY_DOUBLE,
    np_longdouble = NPY_LONGDOUBLE,
    np_cfloat = NPY_CFLOAT,
    np_cdouble = NPY_CDOUBLE,
    np_clongdouble = NPY_CLONGDOUBLE,
    np_object = NPY_OBJECT,
    
    np_int8 = NPY_INT8,
    np_int16 = NPY_INT16,
    np_int64 = NPY_INT64,
    np_uint8 = NPY_UINT8,
    np_uint16 = NPY_UINT16,
    np_uint64 = NPY_UINT64,
    
    np_float32 = NPY_FLOAT32,
    np_float64 = NPY_FLOAT64,
    
    np_complex64 = NPY_COMPLEX64,
    np_complex128 = NPY_COMPLEX128
};
#endif
