#ifndef EIGEN_AUX_HH
#define EIGEN_AUX_HH

#include "utils.hh"

namespace Eigen {

int fit_poly(int argc, char **argv);
int eval_poly(int argc, char **argv);


// structure for storing fitted polynom coefficients
struct poly_fit {
    double mean_t, start_t, stop_t;
    double *mean_coords, *coeffs;
    utils::uint is_centered, deg;
    
    poly_fit() {};
    
    poly_fit(double _mean_t, double _start_t, double _stop_t,
             double * _mean_coords, double * _coeffs, utils::uint _is_centered,
             utils::uint _deg);
};

bool read_poly_fit(const char * filename, poly_fit& fit);


} // Eigen
#endif
