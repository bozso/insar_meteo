#ifndef ARRAYIO_H
#define ARRAYIO_H

#include "common.h"

extern_begin

typedef enum dtype {
    dt_size_t,
    dt_char,
    dt_dtype,
    dt_bool,
    dt_byte,
    dt_ubyte,
    dt_short,
    dt_ushort,
    dt_int,
    dt_uint,
    dt_long,
    dt_ulong,
    dt_longlong,
    dt_ulonglong,

    dt_double,
    dt_longdouble,
    dt_cfloat,
    
    dt_int8,
    dt_int16,
    dt_int64,
    dt_uint8,
    dt_uint16,
    dt_uint64,
    
    dt_float32,
    dt_float64,
    
    dt_complex64,
    dt_complex128
} dtype;


struct _array {
    dtype type;
    size_t ndim, ndata, datasize, *shape, *stride;
    void *data;
    dtor dtor_;
};

typedef struct _array* arrayptr;

arrayptr array_new(dtype const type, size_t const ndim,
                   char const layout, size_t const* shape);

arrayptr array_read(char const* path);
int array_write(arrayptr const arr, char const* path, char const* doc);

extern_end

#endif
