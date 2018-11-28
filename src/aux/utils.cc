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


thread_local static struct _Pool {
    unsigned char *mem, *ptr;
    size_t refcount, storage_size;
} Pool = {0};


void incref(void) { Pool.refcount++; }

void decref(void)
{
    if (--(Pool.refcount) == 0)  {
        Pool.storage_size = 0;
        PyMem_Del(Pool.mem);
        Pool.mem = Pool.ptr = NULL;
    }
}


size_t getref(void) { return Pool.refcount; }
size_t getsize(void) { return Pool.storage_size; }


void init_pool(size_t num)
{
    Pool.storage_size = num;
    Pool.refcount = 0;
    
    if ((Pool.mem = PyMem_New(unsigned char, num)) == NULL) {
        // raise Exception
    }
    Pool.ptr = Pool.mem;
}


void init_pool(int num, ...)
{
    size_t storage_size = 0;
    va_list vl;
    va_start(vl, num);
    
    for(size_t ii = num; ii--;)
        storage_size += va_arg(vl, size_t);
    
    va_end(vl);
    
    init_pool(storage_size);
}


void *alloc(size_t num_bytes)
{
    Pool.ptr += num_bytes;
    return (void *) (Pool.ptr - num_bytes);
}


void reset_pool(void)
{
    if (Pool.storage_size) {
        Pool.refcount = 0;
        PyMem_Del(Pool.mem);
        Pool.mem = Pool.ptr = NULL;
    }
}


File::~File()
{
    if (_file) {
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


#if 0
void println(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_FormatStdout(fmt "\n", ap);
    va_end(ap);
}


void errorln(char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_WriteStderr(fmt "\n", ap);
    va_end(ap);
}


void perrorln(char const* perror_str, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    PySys_WriteStderr(fmt "\n", ap);
    va_end(ap);
    perror(perror_str);
}
#endif
