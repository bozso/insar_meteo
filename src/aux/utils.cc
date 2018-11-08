/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdarg.h>

#include "Python.h"
#include "utils.hh"


Pool::Pool(int num, ...)
{
    storage = NULL;
    ptr = NULL;
    storage_size = 0;
    va_list vl;
    
    va_start(vl, num);
    
    for(size_t ii = 0; ii < num; ++ii)
        storage_size += va_arg(vl, size_t);
    
    va_end(vl);
}


Pool::~Pool() {
    PyMem_Del(storage);
    storage = NULL;
}


bool Pool::init()
{
    if ((storage = PyMem_New(unsigned char, storage_size)) == NULL) {
        // raise Exception
        return true;
    }
    
    return false;
}


void * Pool::alloc(size_t num_bytes)
{
    ptr += num_bytes;
    return (void *) (ptr - num_bytes);
}


File::~File()
{
    if (_file != NULL) {
        fclose(_file);
        _file = NULL;
    }
}


bool File::open(char const* path, char const* mode)
{
    if ((_file = fopen(path, mode)) == NULL) {
        perrorln("open", "Failed to open file \"%s\"", path);
        return true;
    }
    return false;
}


void File::close()
{
    fclose(_file);
    _file = NULL;
}


int File::read(char const* fmt, ...) const
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(_file, fmt, ap);
    va_end(ap);
    return ret;
}


int File::write(char const* fmt, ...) const
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(_file, fmt, ap);
    va_end(ap);
    return ret;
}

int File::read(size_t const size, size_t const num, void *var) const {
    return fread(var, size, num, _file);
}


int File::write(size_t const size, size_t const num, void const* var) const {
    return fwrite(var, size, num, _file);
}


void print(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_FormatStdout(fmt, ap);
    va_end(ap);
}


void println(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_FormatStdout(fmt"\n", ap);
    va_end(ap);
}


void error(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_WriteStderr(fmt, ap);
    va_end(ap);
}


void errorln(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_WriteStderr(fmt"\n", ap);
    va_end(ap);
}


void perrorln(char const* perror_str, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_WriteStderr(fmt"\n", ap);
    va_end(ap);
    perror(perror_str);
}

