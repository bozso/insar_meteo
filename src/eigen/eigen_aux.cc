
namespace Eigen {

poly_fit::poly_fit(double _mean_t, double _start_t, double _stop_t,
                   double * _mean_coords, double * _coeffs, uint _is_centered,
                   uint _deg)
    {
        mean_t = _mean_t;
        start_t = _start_t;
        stop_t = _stop_t;
        
        mean_coords = _mean_coords;
        coeffs = _coeffs;
        
        is_centered = _is_centered;
        deg = _deg;
    }

bool read_poly_fit(const char * filename, poly_fit& fit)
{
    
    
}

} // Eigen
