#ifndef MATH_AUX_HH
#define MATH_AUX_HH

#include "utils.hh"

int fit_poly(int argc, char **argv);
int eval_poly(int argc, char **argv);

// structure for storing fitted polynom coefficients
typedef struct poly_fit_t {
    double mean_t, start_t, stop_t;
    double *mean_coords, *coeffs;
    uint is_centered, deg;
} poly_fit;

bool read_poly_fit(const char * filename, poly_fit *fit);

#endif
