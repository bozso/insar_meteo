/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "insar_functions.hpp"

namespace py = pybind11;
using namespace std;

static void calc_pos(const orbit_fit& orb, double time, double& X, double& Y, double& Z)
{
    // Calculate satellite position based on fitted polynomial orbits at time
    
    uint n_poly = orb.deg + 1, is_centered = orb.is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    const double *coeffs = orb.coeffs, *mean_coords = orb.mean_coords;
    
    if (is_centered)
        time -= orb.mean_t;
    
    if(n_poly == 2) {
        x = coeffs[0] * time + coeffs[1];
        y = coeffs[2] * time + coeffs[3];
        z = coeffs[4] * time + coeffs[5];
    }
    else {
        x = coeffs[         0] * time;
        y = coeffs[    n_poly] * time;
        z = coeffs[2 * n_poly] * time;

        FOR(ii, 1, n_poly - 1) {
            x = (x + coeffs[             ii]) * time;
            y = (y + coeffs[    n_poly + ii]) * time;
            z = (z + coeffs[2 * n_poly + ii]) * time;
        }
        
        x += coeffs[    n_poly - 1];
        y += coeffs[2 * n_poly - 1];
        z += coeffs[3 * n_poly - 1];
    }
    
    if (is_centered) {
        x += mean_coords[0];
        y += mean_coords[1];
        z += mean_coords[2];
    }
    
    X = x; Y = y; Z = z;
} // end calc_pos

static double dot_product(const orbit_fit& orb, cdouble X, cdouble Y,
                          cdouble Z, double time)
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb.deg + 1;
    
    const double *coeffs = orb.coeffs, *mean_coords = orb.mean_coords;
    
    if (orb.is_centered)
        time -= orb.mean_t;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs[0] * time + coeffs[1];
        sat_y = coeffs[2] * time + coeffs[3];
        sat_z = coeffs[4] * time + coeffs[5];
        vel_x = coeffs[0]; vel_y = coeffs[2]; vel_z = coeffs[4];
    }
    // evaluation of polynom with Horner's method
    else {
        
        sat_x = coeffs[         0] * time;
        sat_y = coeffs[    n_poly] * time;
        sat_z = coeffs[2 * n_poly] * time;

        FOR(ii, 1, n_poly - 1) {
            sat_x = (sat_x + coeffs[             ii]) * time;
            sat_y = (sat_y + coeffs[    n_poly + ii]) * time;
            sat_z = (sat_z + coeffs[2 * n_poly + ii]) * time;
        }
        
        sat_x += coeffs[    n_poly - 1];
        sat_y += coeffs[2 * n_poly - 1];
        sat_z += coeffs[3 * n_poly - 1];
        
        vel_x = coeffs[    n_poly - 2];
        vel_y = coeffs[2 * n_poly - 2];
        vel_z = coeffs[3 * n_poly - 2];
        
        FOR(ii, 0, n_poly - 3) {
            power = static_cast<double>(n_poly - 1.0 - ii);
            vel_x += ii * coeffs[             ii] * pow(time, power);
            vel_y += ii * coeffs[    n_poly + ii] * pow(time, power);
            vel_z += ii * coeffs[2 * n_poly + ii] * pow(time, power);
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
    
    return((vel_x * dx  + vel_y * dy  + vel_z * dz) * inorm);
}
// end dot_product

static void closest_appr(const orbit_fit& orb, cdouble X, cdouble Y,
                         cdouble Z, const uint max_iter, double& x, double& y, double& z)
{
    // Compute the sat position using closest approche.
    
    // first, last and middle time, extending the time window by 5 seconds
    double t_start = orb.start_t - 5.0,
           t_stop  = orb.stop_t + 5.0,
           t_middle;
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while( abs(dot_middle) > 1.0e-11 && itr < max_iter) {
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

        ++itr;
    }
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, x, y, z);
} // end closest_appr

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

py::array_t<double> azi_inc (double t_start, double t_stop, double t_mean,
                             uint is_centered, uint max_iter, uint is_lonlat,
                             py::array_t<double>coeffs,
                             py::array_t<double> coords,
                             py::array_t<double> mean_coords)
{
    auto buf_coeffs = coeffs.request(), buf_mean_coords = mean_coords.request();
    
    if (buf_coeffs.ndim != 2)
        throw runtime_error("Number of dimension for coeffs must be 2!");
    
    
    if (buf_mean_coords.ndim != 1)
        throw runtime_error("Number of dimension for mean_coords must be 1!");
    
    orbit_fit orb;
    
    orb.deg = buf_coeffs.shape[1];
    orb.is_centered = is_centered;
    orb.coeffs = (double *) buf_coeffs.ptr;
    orb.mean_coords = (double *) buf_mean_coords.ptr;
    
    // topocentric parameters in PS local system
    double xf, yf, zf,
           xl, yl, zl,
           X, Y, Z,
           t0, lon, lat, h,
           azi, inc;
    
    auto rcoords = coords.unchecked<2>();
    
    auto result = py::array_t<double>(rcoords.size());

#if 0
    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */
    
    np_check_ndim(a_coeffs, 2);
    np_check_dim(a_coeffs, 0, 3);
    np_check_dim(a_coeffs, 1, deg + 1);
    
    // should be nx3 matrix
    np_check_ndim(a_coords, 2);
    np_check_dim(a_coords, 1, 3);
    
    n_coords = np_dim(a_coords, 0);
    
    // should be a 3 element vector
    np_check_ndim(a_meancoords, 1);
    np_check_dim(a_meancoords, 0, 3);
    
    azi_inc_shape[0] = n_coords;
    azi_inc_shape[1] = 2;
    
    // matrix holding azimuth and inclinations values
    np_empty(azi_inc, 2, azi_inc_shape, NPY_DOUBLE, 0);
    
    // Set up orbit polynomial structure
    orb.coeffs = (double *) np_data(a_coeffs);
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.start_t = start_t;
    orb.stop_t = stop_t;
    orb.mean_t = mean_t;
    
    orb.mean_coords = (double *) np_data(a_meancoords);
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, n_coords) {
            lon = np_delem(a_coords, ii, 0) * DEG2RAD;
            lat = np_delem(a_coords, ii, 1) * DEG2RAD;
            h   = np_delem(a_coords, ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         (double *) np_gptr(azi_inc, ii, 0),
                         (double *) np_gptr(azi_inc, ii, 1));
            
        }
        // end for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, n_coords) {
            X = np_delem(a_coords, ii, 0);
            Y = np_delem(a_coords, ii, 1);
            Z = np_delem(a_coords, ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         (double *) np_gptr(azi_inc, ii, 0),
                         (double *) np_gptr(azi_inc, ii, 1));
        }
        // end for
    }
    // end else
#endif
    return result;
}

void test(void)
{
    double x, y, z, lon, lat, height;
    lon = 45.0 * DEG2RAD; lat = 25.0 * DEG2RAD; height = 63900.0;

    FOR(ii, 0, 10000000) {
        ell_cart(lon, lat, height, x, y, z);
        cart_ell(x, y, z, lon, lat, height);
    }
    
    py::print(lon * RAD2DEG, lat * RAD2DEG, height);
}


PYBIND11_MODULE(insar_aux, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test", &test, "test");
    m.def("azi_inc", &azi_inc, "azi_inc");
}
