#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#include "File.h"
#include "common.h"
#include "utils.h"

extern_begin

static void
dtor_(void *obj);


File
open(char const* path, char const* mode)
{
    File *new;
    
    if ((new = Mem_New(File, 1)) == NULL) {
        return NULL;
    }
    
    if ((new->_file = fopen(path, mode)) == NULL) {
        Perror("open", "Could not open file: %s\n", path);
        Mem_Free(new);
        return NULL;
    }
    
    new->dtor_ = &dtor_;

    return new;
}


static void
dtor_(void *obj)
{
    fclose(((File *)obj)->_file);
    ((File *)obj)->_file = NULL;
}


int
write(File const file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(file->_file, fmt, ap);
    va_end(ap);
    return ret;
}

int
read(File const file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(file->_file, fmt, ap);
    va_end(ap);
    return ret;
}

size_t
writeb(File const file, size_t const size, size_t const count, void const* data)
{
    return fwrite(data, size, count, file->_file);
}

size_t
readb(File const file, size_t const size, size_t const count, void const* data)
{
    return fread(data, size, count, file->_file);
}


extern_end
