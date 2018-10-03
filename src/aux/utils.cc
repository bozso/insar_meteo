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

#include "utils.hh"

#define min_arg 2

bool main_check_narg(const int argc, const char * Modules)
{
    if (argc < 2) {
        error("\nAt least one argument, the module name, is required.\
                 \n\nModules to choose from: %s.\n\n", Modules);
        error("Use --help or -h as the first argument to print the help message.\n");
        return true;
    }
    return false;
}


bool check_narg(const argparse& ap, int req_arg)
{
    if (ap.argc == 2) {
        errorln("\n Required number of arguments is %d, current number of "
                "arguments: %d!\n", req_arg, ap.argc - min_arg);
        printf("%s", ap.usage);
        return true;
    }

    const char *first_arg = ap.argv[2];
    
    if (str_equal(first_arg, "-h") or str_equal(first_arg, "--help")) {
        printf("%s", ap.usage);
        return true;
    }
    else if (ap.argc != (req_arg + min_arg)) {
        errorln("\n Required number of arguments is %d, current number of "
                "arguments: %d!\n", req_arg, ap.argc - min_arg);
        printf("%s", ap.usage);
        return true;
    }
    return false;
};

bool get_arg(const argparse& ap, const uint idx, const char * fmt, void *target)
{
    if (sscanf(ap.argv[1 + idx], fmt, target) != 1) {
        errorln("Invalid argument: %s", ap.argv[1 + idx]);
        printf("%s", ap.usage);
        return true;
    }
    return false;
}


bool open(File& file, const char * path, const char * mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        perrorln("open", "Failed to open file \"%s\"", path);
        return true;
    }
    return false;
}

void close(File& file) {
    fclose(file._file);
    file._file = NULL;
}


int read(const File& file, const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(file._file, fmt, ap);
    va_end(ap);
    return ret;
}


int write(const File& file, const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(file._file, fmt, ap);
    va_end(ap);
    return ret;
}

int read(const File& file, const size_t size, const size_t num, void *var)
{
    return fread(var, size, num, file._file);
}

int write(const File& file, const size_t size, const size_t num, const void *var)
{
    return fwrite(var, size, num, file._file);
}


void println(const char * fmt, ...)
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

void perrorln(const char * perror_str, const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap), fputs("\n", stderr);
    va_end(ap);
    perror(perror_str);
}


void error(const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}
