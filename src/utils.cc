#include <stdarg.h>

#include "utils.hh"


bool open(File& file, const char * path, const char * mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("Error");
        return true;
    }
    return false;
}

bool write(File& file, const char * fmt, ...)
{
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(tmp, fmt, ap);
    va_end(ap);
}

bool read(File& file, const char * fmt, ...)
{
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    vfscanf(tmp, fmt, ap);
    va_end(ap);
}

#if 0

bool error(const char * fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

#endif
