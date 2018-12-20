#include <stdlib.h>
#include <stdarg.h>

#include "File.h"
#include "common.h"
#include "utils.h"

static void
dtor_(void *obj);

File *
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

