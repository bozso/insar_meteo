#ifndef EIGEN_AUX_HH
#define EIGEN_AUX_HH

namespace Eigen {

// structure for storing fitted polynom coefficients
struct poly_fit {
    double mean_t, start_t, stop_t;
    double *mean_coords, *coeffs;
    uint is_centered, deg;
    
    poly_fit() {};
    
    poly_fit(double _mean_t, double _start_t, double _stop_t,
              double * _mean_coords, double * _coeffs, uint _is_centered,
              uint _deg);
};

bool read_poly_fit(const char * filename, poly_fit& fit);


} // Eigen
#endif
