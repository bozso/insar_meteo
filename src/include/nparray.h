#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>
#include "common.h"

extern_begin

#include "numpy/ndarrayobject.h"

struct nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    dtor dtor_;
}


typedef enum _newtype {
    empty,
    zeros
} newtype;


typedef enum _dtype {
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
} dtype;

extern_end

#endif
