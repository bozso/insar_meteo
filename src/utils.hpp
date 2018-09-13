#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

typedef unsigned int uint;
typedef const unsigned int cuint;
typedef const double cdouble;

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

/*************
 * IO macros *
 *************/

#define error(text, ...) fprintf(stderr, text, __VA_ARGS__)
#define errors(text) fprintf(stderr, text)
#define errorln(text, ...) fprintf(stderr, text"\n", __VA_ARGS__)

#define print(string, ...) printf(string, __VA_ARGS__)
#define println(format, ...) printf(format"\n", __VA_ARGS__)

#define aux_checkarg(num, doc)\
do {\
    if (argc != ((num) + min_arg)) {\
        errorln("\n Required number of arguments is %d, current number of arguments: %d!",\
                 (num), argc - min_arg);\
        printf((doc));\
        return err_arg;\
    }\
} while(0)

int aux_open()

#define aux_open(file, path, mode)\
do {\
    if (((file) = fopen((path), (mode))) == NULL)\
    {\
        errorln("FILE: %s, LINE: %d :: Failed to open file %s!", __FILE__, __LINE__, path);\
        perror("Error");\
        return errno;\
    }\
} while(0)

#endif // UTILS_HPP
