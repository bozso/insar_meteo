#ifndef UTILS_H
#define UTILS_H

#include "common.h"

extern_begin


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

void error(char const* fmt, ...);
void Perror(char const* perror_str, char const* fmt, ...);


/**************
 * for macros *
 **************/

#define FOR(ii, max) for(size_t (ii) = (max); (ii)--; )
#define FORZ(ii, max) for(size_t (ii) = 0; (ii) < max; ++(ii))
#define FORS(ii, min, max, step) for(size_t (ii) = (min); (ii) < (max); (ii) += (step))
#define FOR1(ii, min, max) for(size_t (ii) = (min); (ii) < (max); ++(ii))


extern_end

#endif
