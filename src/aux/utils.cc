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


File::~File()
{
    if (_file != NULL) {
        fclose(_file);
        _file = NULL;
    }
}


bool open(File& file, char const* path, char const* mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        perrorln("open", "Failed to open file \"%s\"", path);
        return true;
    }
    return false;
}


void close(File& file)
{
    fclose(file._file);
    file._file = NULL;
}


int read(const File& file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(file._file, fmt, ap);
    va_end(ap);
    return ret;
}


int write(const File& file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(file._file, fmt, ap);
    va_end(ap);
    return ret;
}

int read(const File& file, size_t const size, size_t const num, void *var)
{
    return fread(var, size, num, file._file);
}

int write(const File& file, size_t const size, size_t const num, void const* var)
{
    return fwrite(var, size, num, file._file);
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

