#include "satorbit.hpp"
#include "math.hpp"
#include "array.hpp"


static aux::ptr<Ellipsoid const> ellipsoid = nullptr;


using namespace consts;

#if 0

static inline double norm(cdouble x, cdouble y, cdouble z)
{
    return sqrt(x * x + y * y + z * z);
}

// Calculate satellite position based on fitted polynomial orbits at time

static inline void calc_pos(fit_poly const& orb, double time, cart& pos)
{
    size_t n_poly = orb.deg + 1, is_centered = orb.is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    View<double> const& coeffs = orb.coeffs;
    double const *mean_coords = orb.mean_coords;
    
    if (is_centered)
        time -= orb.mean_t;
    
    if(n_poly == 2) {
        x = coeffs(0, 0) * time + coeffs(0, 1);
        y = coeffs(1, 0) * time + coeffs(1, 1);
        z = coeffs(2, 0) * time + coeffs(2, 1);
    }
    else {
        x = coeffs(0, 0)  * time;
        y = coeffs(1, 0)  * time;
        z = coeffs(2, 0)  * time;

        m_for1(ii, 1, n_poly - 1) {
            x = (x + coeffs(0, ii)) * time;
            y = (y + coeffs(1, ii)) * time;
            z = (z + coeffs(2, ii)) * time;
        }
        
        x += coeffs(0, n_poly - 1);
        y += coeffs(1, n_poly - 1);
        z += coeffs(2, n_poly - 1);
    }
    
    if (is_centered) {
        x += mean_coords[0];
        y += mean_coords[1];
        z += mean_coords[2];
    }
    
    pos.x = x; pos.y = y; pos.z = z;
} // calc_pos


static inline double dot_product(fit_poly const& orb, cdouble X, cdouble Y,
                                 cdouble Z, double time)
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    size_t n_poly = orb.deg + 1;
    
    View<double> const& coeffs = orb.coeffs;
    double const *mean_coords = orb.mean_coords;
    
    if (orb.is_centered)
        time -= orb.mean_t;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs(0, 0) * time + coeffs(0, 1);
        sat_y = coeffs(1, 0) * time + coeffs(1, 1);
        sat_z = coeffs(2, 0) * time + coeffs(2, 1);
        
        vel_x = coeffs(0, 0);
        vel_y = coeffs(1, 0);
        vel_z = coeffs(2, 0);
    }
    // evaluation of polynom with Horner's method
    else {
        sat_x = coeffs(0, 0)  * time;
        sat_y = coeffs(1, 0)  * time;
        sat_z = coeffs(2, 0)  * time;

        m_for1(ii, 1, n_poly - 1) {
            sat_x = (sat_x + coeffs(0, ii)) * time;
            sat_y = (sat_y + coeffs(1, ii)) * time;
            sat_z = (sat_z + coeffs(2, ii)) * time;
        }
        
        sat_x += coeffs(0, n_poly - 1);
        sat_y += coeffs(1, n_poly - 1);
        sat_z += coeffs(2, n_poly - 1);

        vel_x = coeffs(0, n_poly - 2);
        vel_y = coeffs(1, n_poly - 2);
        vel_z = coeffs(2, n_poly - 2);
        
        m_for1(ii, 0, n_poly - 3) {
            power = double(n_poly - 1.0 - ii);
            vel_x += ii * coeffs(0, ii) * pow(time, power);
            vel_y += ii * coeffs(1, ii) * pow(time, power);
            vel_z += ii * coeffs(2, ii) * pow(time, power);
        }
    }
    
    if (orb.is_centered) {
        sat_x += mean_coords[0];
        sat_y += mean_coords[1];
        sat_z += mean_coords[2];
    }
    
    // satellite coordinates - GNSS coordinates
    dx = sat_x - X;
    dy = sat_y - Y;
    dz = sat_z - Z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return (vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm;
} // dot_product


// Compute the sat position using closest approche.
static inline void closest_appr(fit_poly const& orb, cdouble X, cdouble Y,
                                cdouble Z, size_t max_iter, cart& sat_pos)
{
    // first, last and middle time, extending the time window by 5 seconds
    double t_start = orb.start_t - 5.0,
           t_stop  = orb.stop_t + 5.0,
           t_middle = 0.0;
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    size_t itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while (abs(dot_middle) > 1.0e-11 and itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, X, Y, Z, t_middle);
        
        // change start for middle
        if ((dot_start * dot_middle) > 0.0) {
            t_start = t_middle;
            dot_start = dot_middle;
        }
        // change  end  for middle
        else
            t_stop = t_middle;

        itr++;
    } // while
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // closest_appr


void ell_cart(cdouble lon, cdouble lat, cdouble h,
              double& x, double& y, double& z)
{
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    x = (              n + h) * cos(lat) * cos(lon);
    y = (              n + h) * cos(lat) * sin(lon);
    z = ( (1.0 - E2) * n + h) * sin(lat);

} // ell_cart


void cart_ell(cdouble x, cdouble y, cdouble z,
              double& lon, double& lat, double& h)
{
    double n, p, o, so, co;

    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n = WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    lat = o;
    
    o = atan(y/x); if(x < 0.0) o += pi;
    lon = o;
    h = p / co - n;
} // cart_ell



static inline void _azi_inc(fit_poly const& orb, cdouble X, cdouble Y,
                            cdouble Z, cdouble lon, cdouble lat,
                            size_t max_iter, double& azi, double& inc)
{
    double xf, yf, zf, xl, yl, zl, t0;
    cart sat;
    
    // satellite closest approache cooridantes
    closest_appr(orb, X, Y, Z, max_iter, sat);
    
    xf = sat.x - X;
    yf = sat.y - Y;
    zf = sat.z - Z;
    
    // estiamtion of azimuth and inclination
    xl = - sin(lat) * cos(lon) * xf
         - sin(lat) * sin(lon) * yf + cos(lat) * zf ;
    
    yl = - sin(lon) * xf + cos(lon) * yf;
    
    zl = + cos(lat) * cos(lon) * xf
         + cos(lat) * sin(lon) * yf + sin(lat) * zf ;
    
    t0 = norm(xl, yl, zl);
    
    inc = acos(zl / t0) * rad2deg;
    
    if(xl == 0.0) xl = 0.000000001;
    
    double temp_azi = atan(abs(yl / xl));
    
    if( (xl < 0.0) && (yl > 0.0) ) temp_azi = consts::pi - temp_azi;
    if( (xl < 0.0) && (yl < 0.0) ) temp_azi = consts::pi + temp_azi;
    if( (xl > 0.0) && (yl < 0.0) ) temp_azi = 2.0 * consts::pi - temp_azi;
    
    temp_azi *= rad2deg;
    
    if(temp_azi > 180.0)
        temp_azi -= 180.0;
    else
        temp_azi += 180.0;
    
    azi = temp_azi;
}
// _azi_inc


void calc_azi_inc(cref<fit_poly>  orb, arr_in coords, arr_out azi_inc,
                  size_t const max_iter, int const is_lonlat)
{
    double X, Y, Z, lon, lat, h;
    X = Y = Z = lon = lat = h = 0.0;
    
    size_t nrows = coords.shape(0);
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        m_for(ii, nrows) {
            lon = coords(ii, 0) * deg2rad;
            lat = coords(ii, 1) * deg2rad;
            h   = coords(ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, X, Y, Z);
            
            _azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                    azi_inc(ii, 0), azi_inc(ii, 1));
            
        } // for
    }
    // coords contains X, Y, Z
    else {
        m_for(ii, nrows) {
            X = coords(ii, 0);
            Y = coords(ii, 1);
            Z = coords(ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, lon, lat, h);
        
            _azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                    azi_inc(ii, 0), azi_inc(ii, 1));
        } // for
    } // if
}
#endif

extern "C" {

void print_ellipsoid()
{
    printf("Ellipsoid in use: a: %lf b: %lf e2: %lf\n",
            ellipsoid->a, ellipsoid->b, ellipsoid->e2);
}


void set_ellipsoid(aux::cptr<Ellipsoid const> new_ellipsoid)
{
    ellipsoid = new_ellipsoid;
}

}
