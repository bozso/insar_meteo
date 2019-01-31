#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdarg.h>

#include "common.hpp"

class File {
    FILE *_file;
    ~File();
    
    public:
        File(char const* path, char const* mode);
        void open(char const* path, char const* mode);

        void write(char const* fmt, ...);
        void read(char const* fmt, ...);

        void write(size_t const size, size_t const count, void const* data);
        void read(size_t const size, size_t const count, void* data);
};

/******************
 * Error messages *
 ******************/

void error(char const* fmt, ...);
void Perror(char const* perror_str, char const* fmt, ...);


#ifdef m_inmet_get_impl

File::File(char const* path, char const* mode)
{
    if ((this->_file = fopen(path, mode)) == nullptr)
    {
        Perror("File", "Could not open file: %s\n", path);
        // TODO: throw
    }
}

File::~File()
{
    fclose(this->_file);
}


void File::open(char const* path, char const* mode)
{
    if ((this->_file = fopen(path, mode)) == nullptr)
    {
        Perror("File", "Could not open file: %s\n", path);
        // TODO: throw
    }
}

void File::Write(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(this->_file, fmt, ap);
    va_end(ap);
    
    
}


void File::read(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(file->_file, fmt, ap);
    va_end(ap);
    
    // TODO: check ret
}


void File::write(size_t const size, size_t const count, void const* data)
{
    fwrite(data, size, count, this->_file);
    // TODO: check
}

void File::read(size_t const size, size_t const count, void* data)
{
    fread(data, size, count, this->_file);
    // TODO: check
}


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

#endif

extern_end

#endif
