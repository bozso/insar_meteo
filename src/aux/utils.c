#include <stdio.h>
#include <stdarg.h>

void error(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    fprintf(stderr, fmt, ap);
    va_end(ap);
}


void Perror(char const* perror_str, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    fprintf(stderr, fmt, ap);
    va_end(ap);
    perror(perror_str);
}
