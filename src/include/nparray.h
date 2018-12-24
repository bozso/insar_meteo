#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>
#include <stdbool.h>
#include "common.h"

extern_begin

#include "numpy/ndarrayobject.h"

struct _nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    dtor dtor_;
};

typedef struct _nparray* nparray;

typedef enum newtype {
    empty,
    zeros
} newtype;


nparray from_otf(int const typenum, size_t const ndim, PyObject* obj);
nparray from_of(size_t const ndim, PyObject *obj);
nparray from_data(int const typenum, void *data, size_t ndim, npy_intp *shape);
nparray newarray(int const typenum, newtype const newt, char const layout,
                 size_t ndim, npy_intp *shape);
void * ar_data(nparray const arr);

bool check_rows(nparray arr, size_t const rows);
bool check_cols(nparray arr, size_t const cols);
bool is_f_cont(nparray arr);
bool is_zero_dim(nparray arr);
bool check_scalar(nparray arr);
bool is_python_number(nparray arr);
bool is_not_swapped(nparray arr);
bool is_byte_swapped(nparray arr);
bool can_cast_to(nparray arr, int const totypenum);

typedef enum dtype {
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
