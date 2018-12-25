#ifndef ARRAYIO_H
#define ARRAYIO_H

#include "common.h"

extern_begin

typedef enum dtype {
    np_bool,

    np_byte,
    np_ubyte,
    np_short,
    np_ushort,
    np_int,
    np_uint,
    np_long,
    np_ulong,
    np_longlong,
    np_ulonglong,

    np_double,
    np_longdouble,
    np_cfloat,
    np_cdouble,
    np_clongdouble,
    
    np_int8,
    np_int16,
    np_int64,
    np_uint8,
    np_uint16,
    np_uint64,
    
    np_float32,
    np_float64,
    
    np_complex64,
    np_complex128
} dtype;


struct _array {
    dtype type;
    size_t ndim, ndata, datasize, *shape, *stride;
    void *data;
};

typedef struct _array* arrayptr;

arrayptr array_read(char const* path);
int array_write(arrayptr const arr, char const* path, char const* doc);

extern_end

#endif
