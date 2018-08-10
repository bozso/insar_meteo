#ifndef SATORBIT_H
#define SATORBIT_H

#include "params_types.h"

/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/
#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); ++(ii))
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))


/************************
 * Structs and typedefs *
 ************************/

// structure for storing fitted orbit polynom coefficients
typedef struct orbit_fit_t {
    double mean_t;
    double * mean_coords;
    double start_t, stop_t;
    double * coeffs;
    uint is_centered, deg;
} orbit_fit;

// cartesian coordinate
typedef struct cart_t { double x, y, z; } cart;

void ell_cart (cdouble lon, cdouble lat, cdouble h,
               double *x, double *y, double *z);

void cart_ell (cdouble x, cdouble y, cdouble z,
               double *lon, double *lat, double *h);

extern void calc_azi_inc(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, cdouble lon, cdouble lat,
                         const uint max_iter, double * azi, double * inc);

#endif
