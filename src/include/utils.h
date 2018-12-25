#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#include "common.h"


extern_begin

struct _File {
    FILE *_file;
    dtor dtor_;
};

typedef struct _File* File;

File
open(char const* path, char const* mode);

int
write(File const file, char const* fmt, ...);

int
read(File const file, char const* fmt, ...);

size_t
writeb(File const file, size_t const size, size_t const count, void const* data)

size_t
readb(File const file, size_t const size, size_t const count, void const* data)


typedef struct argp {
    int argc, req;
    char **argv;
    char const *module;
    char const *doc;
} argp;


int check_narg(argp const *ap);
int get_arg(argp const *ap, int const pos, char const *fmt, void *par)


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

#endif
