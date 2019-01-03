#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdarg.h>

#include "common.h"
#include "utils.h"

extern_begin

struct _File {
    FILE *_file;
    dtor dtor_;
};

typedef struct _File* File;

File
open(char const* path, char const* mode);

int
Write(File const file, char const* fmt, ...);

int
Read(File const file, char const* fmt, ...);

size_t
Writeb(File const file, size_t const size, size_t const count, void const* data);

size_t Readb(File file, size_t const size, size_t const count, void* data);


typedef struct argp {
    int argc, req;
    char **argv;
    char const *module_name;
    char const *doc;
} argp;


int check_narg(argp const *ap);
int get_arg(argp const *ap, int const pos, char const *fmt, void *par);


/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
static const double R_earth = 6372000.0;

static const double WA = 6378137.0;
static const double WB = 6356752.3142;

// (WA*WA-WB*WB)/WA/WA
static const double E2 = 6.694380e-03;


/********************
 * Useful constants *
 ********************/

static double const pi = 3.14159265358979;
static double const pi_per_4 = 3.14159265358979 / 4.0;

static const double deg2rad = 1.745329e-02;
static const double rad2deg = 5.729578e+01;


/******************
 * Error messages *
 ******************/

void error(char const* fmt, ...);
void Perror(char const* perror_str, char const* fmt, ...);


extern_end

#ifdef m_inmet_get_impl

extern_begin


static void File_dtor(void *obj);

File open(char const* path, char const* mode)
{
    File new;
    
    if ((new = Mem_New(struct _File, 1)) == NULL) {
        return NULL;
    }
    
    if ((new->_file = fopen(path, mode)) == NULL) {
        Perror("open", "Could not open file: %s\n", path);
        Mem_Free(new);
        return NULL;
    }
    
    new->dtor_ = &File_dtor;

    return new;
}


static void File_dtor(void *obj)
{
    fclose(((File)obj)->_file);
    ((File)obj)->_file = NULL;
}


int Write(File const file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfprintf(file->_file, fmt, ap);
    va_end(ap);
    return ret;
}

int Read(File const file, char const* fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    int ret = vfscanf(file->_file, fmt, ap);
    va_end(ap);
    return ret;
}

size_t Writeb(File const file, size_t const size, size_t const count, void const* data)
{
    return fwrite(data, size, count, file->_file);
}

size_t Readb(File file, size_t const size, size_t const count, void* data)
{
    return fread(data, size, count, file->_file);
}


static void print_usage(argp const *ap)
{
    error("Usage: inmet %s %s", ap->module_name, ap->doc);
}


int check_narg(argp const *ap)
{
    if (ap->argc != ap->req) {
        error("Required number of arguments is %d, but got %d number of "
              "arguments!\n");
        print_usage(ap);
        return 1;
    }
    return 0;
}


int get_arg(argp const *ap, int const pos, char const *fmt, void *par)
{
    char const *buffer = ap->argv[pos];
    
    if (sscanf(buffer, fmt, par) <= 0) {
        error("Failed to parse argument %s!\n", buffer);
        print_usage(ap);
        return 1;
    }
    return 0;
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
