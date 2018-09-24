#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#include "utils.hh"

namespace utils {

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

/****************
 * IO functions *
 ****************/

void error(const char * fmt, ...);
void errorln(const char * fmt, ...);

void println(const char * fmt, ...);

/********************
 * Argument parsing *
 ********************/

struct argparse {
    int argc;
    char **argv;
    const char * usage;
    
    argparse(int _argc, char **_argv, const char * _usage):
        argc(_argc), argv(_argv), usage(_usage) {};
    
    void print_usage() const
    {
        printf("Usage: \n%s\n", usage);
    }
};

bool check_narg(const argparse& ap, int req_arg);

template<typename T>
bool get_arg(const argparse& ap, const uint idx, const char * fmt, T& target)
{
    if (sscanf(ap.argv[1 + idx], fmt, &target) != 1) {
        errorln("Invalid argument: %s", ap.argv[1 + idx]);
        ap.print_usage();
        return true;
    }
    return false;
}


struct File {
    FILE * _file;
    
    File()
    {
        _file = NULL;
    }
    
    ~File()
    {
        if (_file != NULL)
        {
            fclose(_file);
            _file = NULL;
        }
    }
};

bool open(File& file, const char * path, const char * mode);

int fprint(File& file, const char * fmt, ...);
int fscan(File& file, const char * fmt, ...);

#define ut_check(condition)\
do {\
    if ((condition))\
        goto fail;\
} while (0)

} // utils

#endif // UTILS_HPP
