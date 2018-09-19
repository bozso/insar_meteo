#include <stdarg.h>

#include "utils.hh"

namespace utils {

bool open(File& file, const char * path, const char * mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("Error");
        return true;
    }
    return false;
}

void println(char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vprintf(fmt, ap), puts("\n");
    va_end(ap);
}

void errorln(const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap), fputs("\n", stderr);
    va_end(ap);
}

void error(const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

int fprint(File& file, const char * fmt, ...)
{
    int ret = 0;
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    ret = vfprintf(tmp, fmt, ap);
    va_end(ap);

    return ret;
}

int fscan(File& file, const char * fmt, ...)
{
    int ret = 0;
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    
    ret = vfscanf(tmp, fmt, ap);
    va_end(ap);
    return ret;
}

}
