#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdbool.h>
#include <iso646.h>

typedef unsigned int uint;
typedef const unsigned int cuint;
typedef const double cdouble;

#define OK 0
#ifndef EIO
#define EIO 1
#endif
#define EARG 2

#define min_arg 2

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA 6378137.0
#define WB 6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2 6.694380e-03


#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

/**************
 * for macros *
 **************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); ++(ii))
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

bool main_check_narg(const int argc, const char * Modules);

/****************
 * IO functions *
 ****************/

void error(const char * fmt, ...);
void errorln(const char * fmt, ...);

void println(const char * fmt, ...);

/********************
 * Argument parsing *
 ********************/

typedef struct argparse_t {
    int argc;
    char **argv;
    const char *usage;
} argparse;

bool check_narg(const argparse * ap, int req_arg);
bool get_arg(const argparse *ap, const uint idx, const char * fmt, void *target);

#define ut_module_select(str) (not strcmp(argv[1], (str)))

#define ut_check(condition)\
do {\
    if ((condition))\
        goto fail;\
} while (0)

#endif // UTILS_HPP
