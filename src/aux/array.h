#ifndef ARRAY_H
#define ARRAY_H

#include <stdint.h>
#include <string.h>
#include <complex.h>

#include "utils.h"
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

    dt_float,
    dt_double,
    dt_longdouble,
    
    dt_int8,
    dt_int16,
    dt_int64,
    dt_uint8,
    dt_uint16,
    dt_uint64,
    
    dt_cfloat,
    dt_cdouble,
    dt_clongdouble
} dtype;


typedef enum layout {
    colmajor,
    rowmajor
} layout;


struct _array {
    dtype type;
    size_t ndim, ndata, datasize, *shape, *stride;
    void *data;
    dtor dtor_;
};


typedef struct _array* arrayptr;

bool array_new(arrayptr* arr, dtype const type, size_t const ndim,
               layout const lay, size_t const* shape);

bool array_read(arrayptr* arr, char const* path);
bool array_write(arrayptr const arr, char const* path, char const* doc);

int get_typenum(char const* name);

#ifdef m_get_impl


static size_t const sizes[] = {
    [dt_size_t]       = sizeof(size_t),
    [dt_char]         = sizeof(char),
    [dt_dtype]        = sizeof(dtype),
    [dt_bool]         = sizeof(unsigned char),
    [dt_byte]         = sizeof(char),
    [dt_ubyte]        = sizeof(unsigned char),
    [dt_short]        = sizeof(short),
    [dt_ushort]       = sizeof(unsigned short),
    [dt_int]          = sizeof(int),
    [dt_uint]         = sizeof(unsigned int),
    [dt_long]         = sizeof(long),
    [dt_ulong]        = sizeof(unsigned long),
    [dt_longlong]     = sizeof(long long int),
    [dt_ulonglong]    = sizeof(unsigned long long int),
    [dt_float]        = sizeof(float),
    [dt_double]       = sizeof(double),
    [dt_longdouble]   = sizeof(long double),
    [dt_int8]         = sizeof(int8_t),
    [dt_int16]        = sizeof(int16_t),
    [dt_int64]        = sizeof(int64_t),
    [dt_uint8]        = sizeof(uint8_t),
    [dt_uint16]       = sizeof(uint16_t),
    [dt_uint64]       = sizeof(uint64_t),
    [dt_cfloat]       = sizeof(float complex),
    [dt_cdouble]      = sizeof(double complex),
    [dt_clongdouble]  = sizeof(double complex)
};


int get_typenum(char const* name)
{
    
    if (str_equal(name, "size_t"))
        return dt_size_t;
    else if (str_equal(name, "char"))
         return dt_char;
    else if (str_equal(name, "dtype"))
         return dt_dtype;

    else if (str_equal(name, "bool"))
         return dt_bool;

    else if (str_equal(name, "byte"))
         return dt_byte;
    else if (str_equal(name, "ubyte"))
         return dt_ubyte;

    else if (str_equal(name, "short"))
         return dt_short;
    else if (str_equal(name, "ushort"))
         return dt_ushort;

    else if (str_equal(name, "int"))
         return dt_int;
    else if (str_equal(name, "uint"))
         return dt_uint;

    else if (str_equal(name, "long"))
         return dt_long;
    else if (str_equal(name, "ulong"))
         return dt_ulong;

    else if (str_equal(name, "longlong"))
         return dt_longlong;
    else if (str_equal(name, "ulonglong"))
         return dt_ulonglong;

    else if (str_equal(name, "float"))
         return dt_float;
    else if (str_equal(name, "double"))
         return dt_double;
    else if (str_equal(name, "longdouble"))
         return dt_longdouble;
    
    else if (str_equal(name, "int8"))
         return dt_int8;
    else if (str_equal(name, "int16"))
         return dt_int16;
    else if (str_equal(name, "int64"))
         return dt_int64;
    
    else if (str_equal(name, "uint8"))
         return dt_uint8;
    else if (str_equal(name, "uint16"))
         return dt_uint16;
    else if (str_equal(name, "uint64"))
         return dt_uint64;

    else if (str_equal(name, "cfloat"))
         return dt_cfloat;
    else if (str_equal(name, "cdouble"))
         return dt_cdouble;
    else if (str_equal(name, "clongdouble"))
         return dt_clongdouble;
    else
        return -1;
}

static void array_dtor(void *arr);

bool array_new(arrayptr* arr, dtype const type, size_t const ndim,
               layout const lay, size_t const* shape)
{
    arrayptr new = Mem_New(struct _array, 1);
    
    if (new == NULL) {
        Perror("array_new", "Memory allocation failed!\n");
        return true;
    }
    
    size_t total = 0;
    
    m_forz(ii, ndim)
        total += shape[ii];
    
    new->datasize = sizes[type];
    new->ndata = total;
    
    if ((new->shape = Mem_New(size_t, 2 * ndim + sizes[type] * total)) == NULL) {
        Perror("array_new", "Memory allocation failed!\n");
        Mem_Free(new);
        return true;
    }
    
    // check memcpy
    new->shape = memcpy(new->shape, shape, ndim * sizes[dt_size_t]);
    
    new->stride = new->shape + ndim;
    
    switch(lay) {
        case rowmajor:
            for(size_t ii = 0; ii < ndim; ++ii) {
                new->stride[ii] = 1;
                for(size_t jj = ii + 1; jj < ndim; ++jj) {
                    new->stride[ii] *= new->shape[jj];
                }
            }
            
            break;

        case colmajor:
            for(size_t ii = 0; ii < ndim; ++ii) {
                new->stride[ii] = 1;
                for(size_t jj = 1; jj < ii - 1; ++jj) {
                    new->stride[ii] *= new->shape[jj];
                }
            }
            
            break;
        //default:
            // error
    }
    
    new->data   = new->shape + 2 * ndim;
    new->dtor_  = array_dtor;
    
    *arr = new;
    
    return false;
}


bool array_read(arrayptr* arr, char const* path)
{
    arrayptr new = NULL;
    File infile = NULL;
    
    if (open(&infile, path, "rb"))
        goto fail;
    
    size_t doc_len = 0;
    char *doc;
    
    if (Readb(infile, sizes[dt_size_t], 1, &doc_len) < 0 or
        Readb(infile, sizes[dt_char], doc_len, doc) < 0)
        goto fail;
    
    size_t ndim = 0, ndata = 0, *shape = NULL;
    dtype type = 0;
    
    if (Readb(infile, sizes[dt_dtype], 1, &type) < 0 or
        Readb(infile, sizes[dt_size_t], 1, &ndim) < 0 or
        Readb(infile, sizes[dt_size_t], 1, &ndata) < 0 or
        Readb(infile, sizes[dt_size_t], ndim, shape) < 0 or)
        goto fail;
    
    if (array_new(&new, type, ndim, ,shape))
        goto fail;
    
    if (Readb(infile, sizes[dt_size_t], ndim, arr->stride) < 0 or
        Readb(infile, arr->datasize, ndata, arr->data) < 0)
        goto fail;
    
    *arr = new;
    del(infile);
    
    return false;

fail:
    del(arr); del(infile);
    return true;
}


bool array_write(arrayptr const arr, char const* path, char const* doc)
{
    File outfile = NULL;
    
    if (open(&outfile, path, "wb"))
        goto fail;
    
    size_t doc_len = strlen(doc);
    
    if(Writeb(outfile, sizes[s_size_t], 1, &doc_len) < 0 or
       Writeb(outfile, sizes[s_char], strlen(doc), doc) < 0)
       goto fail;
    
    size_t ndim = arr->ndim, ndata = arr->ndata;
    
    if (Writeb(outfile, sizes[dt_dtype], 1, &(arr->type)) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndim) < 0)
        goto fail;
        
    if (Writeb(outfile, sizes[dt_dtype], 1, &(arr->type)) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndim) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndata) < 0 or
        Writeb(outfile, sizes[dt_size_t], ndim, arr->shape) < 0 or
        Writeb(outfile, sizes[dt_size_t], ndim, arr->stride) < 0 or
        Writeb(outfile, arr->datasize, ndata, arr->data) < 0)
        goto fail;
    
    del(outfile);
    return false;
    
fail:
    del(outfile);
    Perror("array_write", "Failed to write array to file: %s\n", path);
    return true;
}


static void array_dtor(void *arr)
{
    Mem_Free(((arrayptr)arr)->shape);
    ((arrayptr)arr)->shape = NULL;
}
#endif

extern_end

#endif
