#include <stdlib.h>
#include <stdarg.h>

#include "File.h"
#include "common.h"

static void
dtor_(void *obj);

File *
open(char const* path, char const* mode)
{
    File *new = Mem_Malloc(File, 1);
    
    if ((new->_file = fopen(path, mode)) == NULL) {
        //perror("open: Could not open file: %s ", path);
        return NULL;
    }
    
    new->dtor_ = &dtor_;

    return new;
}


int
write(File const *file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(file->_file, fmt, ap);
    va_end(ap);
    return ret;
}


static void
dtor_(void *obj)
{
    fclose(((File *)obj)->_file);
    ((File *)obj)->_file = NULL;
}

