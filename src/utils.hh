#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

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
