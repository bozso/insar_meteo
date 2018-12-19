#include <stdlib.h>
#include <stdarg.h>
#include "File.h"

static void
File_free(void *obj);

File
open(char const* path, char const* mode)
{
    File tmp = (File){fopen(path, mode), NULL};
    
    tmp.refc = calloc(1, sizeof(ref));
    tmp.refc->obj = &tmp;
    tmp.refc->count = 1;
    tmp.refc->free = &File_free;
    
    return tmp;
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
File_free(void *obj)
{
    fclose(((File *)obj)->_file);
    ((File *)obj)->_file = NULL;
}

