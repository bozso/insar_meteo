#include "File.h"
#include "utils.h"
#include "arrayio.h"

extern_begin

enum size_idx {
    dt_
};


static size_t const sizes[] = {
    [dt_size_t]       = sizeof(size_t),
    [dt_char]         = sizeof(char),
    [dt_dtype]        = sizeof(dtype)
    [dt_bool]         = sizeof(unsigned char),
    [dt_byte]         = sizeof(char),
    [dt_ubyte]        = sizeof(unsigned char),
    [dt_short]        = sizeof(short),
    [dt_ushort]       = sizeof(unsigned short),
    [dt_int]          = sizeof(int),
    [dt_uint]         = sizeof(unsigned int),
    [dt_long]         = sizeof(long int),
    [dt_ulong]        = sizeof(unsigned long int),
    [dt_longlong]     = sizeof(long long int),
    [dt_ulonglong]    = sizeof(unsigned long long int),
    [dt_double]       = sizeof(double),
    [dt_longdouble]   = sizeof(long double),
    [dt_cfloat]       = sizeof(float),
    [dt_int8]         = sizeof(int8_t),
    [dt_int16]        = sizeof(int16_t),
    [dt_int64]        = sizeof(int64_t),
    [dt_uint8]        = sizeof(uint8_t),
    [dt_uint16]       = sizeof(uint16_t),
    [dt_uint64]       = sizeof(uint64_t),
    [dt_float32]      = sizeof(float),
    [dt_float64]      = sizeof(double),
    [dt_complex64]    = sizeof(float complex),
    [dt_complex128]   = sizeof(double complex)
};

static void array_dtor(void *arr);

arrayptr array_new(dtype const type, size_t const ndim,
                   char const layout, size_t const* shape)
{
    arrayptr new = Mem_New(struct _array, 1);
    
    if (new == NULL) {
        Perror("array_new", "Memory allocation failed!\n");
        return NULL;
    }
    
    size_t total = 0;
    
    FORZ(ii, ndim)
        total += shape[ii];
    
    new->datasize = sizes[type];
    new->ndata = total;
    
    
    if ((new->shape = Mem_New(size_t, 2 * ndim + sizes[type] * total)) == NULL)
        Mem_Free(new);
        Perror("array_new", "Memory allocation failed!\n");
        return NULL;
    }
    
    new->stride = new->shape + ndim;
    new->data = new->shape + 2 * ndim
    new->dtor_ = array_dtor;
    
    return new;
}


arrayptr array_read(char const* path)
{
    File infile = open(path, "rb");
    
    if (outfile == NULL)
        return 1;
    
    
    size_t doc_len = 0;
    
    m_check_fail(readb(outfile, sizes[s_size_t], 1, &doc_len) < 0);
    m_check_fail(readb(outfile, sizes[s_char], doc_len, doc) < 0);
    
    size_t ndim = 0, ndata = 0, *shape = NULL;
    dtype type = 0;
    
    m_check_fail(readb(infile, sizes[dt_dtype], 1, &type) < 0);
    m_check_fail(readb(infile, sizes[dt_size_t], 1, &ndim) < 0);
    m_check_fail(readb(infile, sizes[dt_size_t], 1, &ndata) < 0);
    m_check_fail(readb(infile, sizes[dt_size_t], ndim, shape) < 0);
    
    arrayptr arr = array_new(type, ndim, ,shape)
    
    if (arr == NULL)
        goto fail;
    
    m_check_fail(readb(infile, sizes[dt_size_t], ndim, arr->stride) < 0);
    m_check_fail(readb(infile, arr->datasize, ndata, arr->data) < 0);
    
    return arr;

fail:
    del(arr);
    return NULL;
}





int array_write(arrayptr const arr, char const* path, char const* doc)
{
    File outfile = open(path, "wb");
    
    if (outfile == NULL)
        return 1;
    
    size_t doc_len = strlen(doc);
    
    m_check_fail(writeb(outfile, sizes[s_size_t], 1, &doc_len) < 0);
    m_check_fail(writeb(outfile, sizes[s_char], strlen(doc), doc) < 0);
    
    size_t ndim = arr->ndim, ndata = arr->ndata;
    
    m_check_fail(writeb(outfile, sizes[dt_dtype], 1, &(arr->type)) < 0);
    m_check_fail(writeb(outfile, sizes[dt_size_t], 1, &ndim) < 0);
    m_check_fail(writeb(outfile, sizes[dt_size_t], 1, &ndata) < 0);
    m_check_fail(writeb(outfile, sizes[dt_size_t], ndim, arr->shape) < 0);
    m_check_fail(writeb(outfile, sizes[dt_size_t], ndim, arr->stride) < 0);
    m_check_fail(writeb(outfile, arr->datasize, ndata, arr->data) < 0);
    
    return 0;
    
fail:
    del(outfile);
    Perror("array_write", "Failed to write array to file: %s\n", path);
    return 1;
}

static void array_dtor(void *arr)
{
    Mem_Free(((arrayptr)arr)->shape);
    ((arrayptr)arr)->shape = NULL;
}

extern_end
