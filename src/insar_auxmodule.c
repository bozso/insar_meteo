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

#include <stdio.h>
#include <tgmath.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "capi_macros.h"

/************************
 * Structs and typedefs *
 ************************/

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;
typedef const double cdouble;
typedef unsigned int uint;

// structure for storing fitted orbit polynom coefficients
typedef struct {
    double mean_t;
    double * mean_coords;
    double start_t, stop_t;
    double * coeffs;
    uint is_centered, deg;
} orbit_fit;

// cartesian coordinate
typedef struct cart_t { double x, y, z; } cart;

/************************
 * Auxilliary functions *
 * **********************/

static inline double norm(cdouble x, cdouble y, cdouble z)
{
    // vector norm
    return sqrt(x * x + y * y + z * z);
}

static void ell_cart (cdouble lon, cdouble lat, cdouble h,
                      double *x, double *y, double *z)
{
    // from ellipsoidal to cartesian coordinates
    
    double n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));;

    *x = (              n + h) * cos(lat) * cos(lon);
    *y = (              n + h) * cos(lat) * sin(lon);
    *z = ( (1.0 - E2) * n + h) * sin(lat);

} // end of ell_cart


static void cart_ell (cdouble x, cdouble y, cdouble z,
                      double *lon, double *lat, double *h)
{
    // from cartesian to ellipsoidal coordinates
    
    double n, p, o, so, co;

    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    *lat = o;
    
    o = atan(y/x); if(x < 0.0) o += M_PI;
    *lon = o;
    *h = p / co - n;
}
// end of cart_ell

static void calc_pos(const orbit_fit * orb, double time, cart * pos)
{
    // Calculate satellite position based on fitted polynomial orbits at time
    
    uint n_poly = orb->deg + 1, is_centered = orb->is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (is_centered)
        time -= orb->mean_t;
    
    if(n_poly == 2) {
        x = coeffs[0] * time + coeffs[1];
        y = coeffs[2] * time + coeffs[3];
        z = coeffs[4] * time + coeffs[5];
    }
    else {
        x = coeffs[0]           * time;
        y = coeffs[n_poly]      * time;
        z = coeffs[2 * n_poly]  * time;

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
    
    pos->x = x; pos->y = y; pos->z = z;
} // end calc_pos

static double dot_product(const orbit_fit * orb, cdouble X, cdouble Y,
                          cdouble Z, double time)
{
    /* Calculate dot product between satellite velocity vector and
     * and vector between ground position and satellite position. */
    
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb->deg + 1;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (orb->is_centered)
        time -= orb->mean_t;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs[0] * time + coeffs[1];
        sat_y = coeffs[2] * time + coeffs[3];
        sat_z = coeffs[4] * time + coeffs[5];
        vel_x = coeffs[0]; vel_y = coeffs[2]; vel_z = coeffs[4];
    }
    // evaluation of polynom with Horner's method
    else {
        
        sat_x = coeffs[0]           * time;
        sat_y = coeffs[n_poly]      * time;
        sat_z = coeffs[2 * n_poly]  * time;

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
            power = (double) n_poly - 1.0 - ii;
            vel_x += ii * coeffs[             ii] * pow(time, power);
            vel_y += ii * coeffs[    n_poly + ii] * pow(time, power);
            vel_z += ii * coeffs[2 * n_poly + ii] * pow(time, power);
        }
    }
    
    if (orb->is_centered) {
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

static void closest_appr(const orbit_fit * orb, cdouble X, cdouble Y,
                         cdouble Z, const uint max_iter, cart * sat_pos)
{
    // Compute the sat position using closest approche.
    
    // first, last and middle time, extending the time window by 5 seconds
    double t_start = orb->start_t - 5.0,
           t_stop  = orb->stop_t + 5.0,
           t_middle;
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
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
    }
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // end closest_appr

static inline void calc_azi_inc(const orbit_fit * orb,
                                cdouble X, cdouble Y, cdouble Z,
                                cdouble lon, cdouble lat,
                                const uint max_iter, double * azi,
                                double * inc)
{
    double xf, yf, zf, xl, yl, zl, t0, temp_azi;
    cart sat;
    // satellite closest approache cooridantes
    closest_appr(orb, X, Y, Z, max_iter, &sat);
    
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
    
    *inc = acos(zl / t0) * RAD2DEG;
    
    if(xl == 0.0) xl = 0.000000001;
    
    temp_azi = atan(fabs(yl / xl));
    
    if( (xl < 0.0) && (yl > 0.0) ) temp_azi = M_PI - temp_azi;
    if( (xl < 0.0) && (yl < 0.0) ) temp_azi = M_PI + temp_azi;
    if( (xl > 0.0) && (yl < 0.0) ) temp_azi = 2.0 * M_PI - temp_azi;
    
    temp_azi *= RAD2DEG;
    
    if(temp_azi > 180.0)
        temp_azi -= 180.0;
    else
        temp_azi += 180.0;
    
    *azi = temp_azi;
}

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

PyFun_Doc(test, "test");

py_ptr test(void)
{
    double x, y, z, lon, lat, height;
    lon = 45.0 * DEG2RAD; lat = 25.0 * DEG2RAD; height = 63900.0;

    FOR(ii, 0, 10000000) {
        ell_cart(lon, lat, height, &x, &y, &z);
        //println("%lf %lf %lf\n", lon, lat, height);
        cart_ell(x, y, z, &lon, &lat, &height);
    }
    
    println("%lf %lf %lf\n", lon * RAD2DEG, lat * RAD2DEG, height);
    
    Py_RETURN_NONE;
}

#if 0

PyFun_Doc(azi_inc, "azi_inc");

py_ptr azi_inc (PyFun_Varargs)
{
    double start_t, stop_t, mean_t;
    py_ptr coeffs, coords, mean_coords;
    np_ptr a_coeffs = NULL, a_coords = NULL, a_meancoords = NULL,
           azi_inc = NULL;
    uint is_centered, deg, max_iter, is_lonlat;

    cart sat;
    orbit_fit orb;
    npy_intp n_coords, azi_inc_shape[2];

    // topocentric parameters in PS local system
    double xf, yf, zf,
           xl, yl, zl,
           X, Y, Z,
           t0, lon, lat, h,
           azi, inc;
    
    PyFun_Parse_Varargs("OdddOIIOII:azi_inc", &coeffs, &start_t, &stop_t,
                        &mean_t, &mean_coords, &is_centered, &deg, &coords,
                        &max_iter, &is_lonlat);

    // Importing arrays
    np_import(a_coeffs, coeffs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    np_import(a_coords, coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    np_import(a_meancoords, mean_coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

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

    // Cleanup and return
    Py_DECREF(a_coeffs);
    Py_DECREF(a_coords);
    Py_DECREF(a_meancoords);
    return Py_BuildValue("O", azi_inc);

fail:
    Py_XDECREF(a_coeffs);
    Py_XDECREF(a_coords);
    Py_XDECREF(a_meancoords);
    Py_XDECREF(azi_inc);
    return NULL;
} // end azim_inc

PyFun_Doc(asc_dsc_select,
"asc_dsc_select");

py_ptr asc_dsc_select(PyFun_Keywords)
{
    py_ptr in_arr1 = NULL, in_arr2 = NULL;
    np_ptr arr1 = NULL, arr2 = NULL;
    npy_double max_sep = 100.0, max_sep_deg;
    
    npy_intp n_arr1, n_arr2;
    uint n_found = 0;
    npy_double dlon, dlat;
    
    char * keywords[] = {"asc", "dsc", "max_sep", NULL};
    
    PyFun_Parse_Keywords(keywords, "OO|d:asc_dsc_select", &in_arr1, &in_arr2,
                                                          &max_sep);
    
    max_sep /= R_earth;
    max_sep_deg = max_sep * max_sep * RAD2DEG * RAD2DEG;

    println("Maximum separation: %6.3lf meters => approx. %E degrees",
            max_sep, max_sep_deg);
    
    np_import(arr1, in_arr1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    np_import(arr2, in_arr2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    n_arr1 = np_dim(arr1, 0);
    n_arr2 = np_dim(arr2, 0);
    
    np_ptr idx = (np_ptr) PyArray_ZEROS(1, &n_arr1, NPY_BOOL, 0);
    
    FOR(ii, 0, n_arr1) {
        FOR(jj, 0, n_arr2) {
            dlon = np_delem(arr1, ii, 0) - np_delem(arr2, jj, 0);
            dlat = np_delem(arr1, ii, 1) - np_delem(arr2, jj, 1);
            
            if ( dlon * dlon + dlat * dlat < max_sep) {
                //*( (npy_bool *) np_ptr1(idx, ii) ) = 1;
                np_belem1(idx, ii) = 1;
                n_found++;
                break;
            }
        }
    }
    
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    return Py_BuildValue("OI", idx, n_found);

fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}
// end asc_dsc_select

#endif

/**********************
 * Python boilerplate *
 **********************/

make_module_table(insar_aux,
    PyFun_Method_Noargs(test))

init_module(insar_aux)

#if 0
static PyMethodDef InsarMethods[] = {
    //PyFun_Method_Varargs(azi_inc),
    PyFun_Method_Noargs(test),
    //PyFun_Method_Keywords(asc_dsc_select),
    {NULL, NULL, 0, NULL}
};

PyFun_Doc(insar_aux,
"INSAR_AUX");

static struct PyModuleDef insarmodule = {
    PyModuleDef_HEAD_INIT, "insar_aux", insar_aux__doc__, -1, InsarMethods
};

PyMODINIT_FUNC
PyInit_insar_aux(void)
{
    import_array();
    return PyModule_Create(&insarmodule);
}
#endif
