#include "File.h"
#include "arrayio.h"

extern_begin

enum size_idx {
    s_size_t,
    s_char,
    s_dtype
};


static size_t const sizes[] = {
    [s_size_t] = sizeof(size_t),
    [s_char] = sizeof(char),
    [s_dtype] = sizeof(dtype)
};


array array_read(char const* path)
{
    
    
    
    
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
    
    m_check_fail(writeb(outfile, sizes[s_size_t], 1, &ndim) < 0);
    m_check_fail(writeb(outfile, sizes[s_size_t], 1, &ndata) < 0);
    m_check_fail(writeb(outfile, sizes[s_size_t], ndim, arr->shape) < 0);
    m_check_fail(writeb(outfile, sizes[s_size_t], ndim, arr->stride) < 0);
    m_check_fail(writeb(outfile, arr->datasize, ndata, arr->data) < 0);
    
    return 0;
    
fail:
    del(outfile);
    Perror("array_write", "Failed to write array to file: %s\n", path);
    return 1;
}



extern_end
