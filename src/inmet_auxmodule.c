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

#include "Python.h"
#include "numpy/arrayobject.h"

#include "capi_functions.h"
#include "satorbit.h"

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

pyfun_doc(test, "test");

py_ptr test(void)
{
    double x, y, z, lon, lat, height;
    lon = 45.0 * DEG2RAD; lat = 25.0 * DEG2RAD; height = 63900.0;

    FOR(ii, 0, 10000000) {
        ell_cart(lon, lat, height, &x, &y, &z);
        cart_ell(x, y, z, &lon, &lat, &height);
    }
    
    println("%lf %lf %lf\n", lon * RAD2DEG, lat * RAD2DEG, height);
    
    Py_RETURN_NONE;
}

pyfun_doc(azi_inc, "azi_inc");

py_ptr azi_inc (py_varargs)
{
    double start_t, stop_t, mean_t;
    uint is_centered, deg, max_iter, is_lonlat;
    py_ptr coeffs, coords, mean_coords;    

    pyfun_parse_varargs("OdddOIIOII:azi_inc", &coeffs, &start_t, &stop_t,
                        &mean_t, &mean_coords, &is_centered, &deg, &coords,
                        &max_iter, &is_lonlat);

    // Importing arrays
    np_ptr a_coeffs = NULL, a_coords = NULL, a_meancoords = NULL;

    np_import_check_double_in(a_coeffs, coeffs, 2, "coeffs");
    np_import_check_double_in(a_coords, coords, 2, "coords");
    np_import_check_double_in(a_meancoords, mean_coords, 1, "mean_coords");

    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */
    
    np_check_matrix(a_coeffs, 3, deg + 1, "coeffs");
    
    // should be nx3 matrix
    np_check_cols(a_coords, 3, "coords");

    // should be a 3 element vector
    np_check_rows(a_meancoords, 3, "coords");
    
    // number of coordinates
    uint n_coords = np_rows(a_coords);

    npy_intp azi_inc_shape[2] = {(npy_intp) n_coords, 2};
    
    // matrix holding azimuth and inclination values
    np_ptr azi_inc = NULL;
    np_empty_double(azi_inc, 2, azi_inc_shape);
    
    // Set up orbit polynomial structure
    orbit_fit orb;
    orb.coeffs = (double *) np_data(a_coeffs);
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.start_t = start_t;
    orb.stop_t = stop_t;
    orb.mean_t = mean_t;
    
    orb.mean_coords = (double *) np_data(a_meancoords);

    double X, Y, Z, lon, lat, h;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, n_coords) {
            lon = np_delem2(a_coords, ii, 0) * DEG2RAD;
            lat = np_delem2(a_coords, ii, 1) * DEG2RAD;
            h   = np_delem2(a_coords, ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         (npy_double *) np_gptr2(azi_inc, ii, 0),
                         (npy_double *) np_gptr2(azi_inc, ii, 1));
            
        }
        // end for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, n_coords) {
            X = np_delem2(a_coords, ii, 0);
            Y = np_delem2(a_coords, ii, 1);
            Z = np_delem2(a_coords, ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         (npy_double *) np_gptr2(azi_inc, ii, 0),
                         (npy_double *) np_gptr2(azi_inc, ii, 1));
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

pyfun_doc(asc_dsc_select,
"asc_dsc_select");

py_ptr asc_dsc_select(py_keywords)
{
    char * keywords[] = {"asc", "dsc", "max_sep", NULL};

    py_ptr in_arr1 = NULL, in_arr2 = NULL;
    npy_double max_sep = 100.0;

    pyfun_parse_keywords(keywords, "OO|d:asc_dsc_select",
                         &in_arr1, &in_arr2, &max_sep);
    
    npy_double max_sep_deg = max_sep / R_earth;
    max_sep_deg = max_sep_deg * max_sep_deg * RAD2DEG * RAD2DEG;

    println("Maximum separation: %6.3lf meters => approx. %E degrees",
            max_sep, max_sep_deg);

    np_ptr arr1 = NULL, arr2 = NULL;

    np_import_check_double_in(arr1, in_arr1, 2, "asc");
    np_import_check_double_in(arr2, in_arr2, 2, "dsc");
    
    uint n_arr1 = np_rows(arr1), n_arr2 = np_rows(arr2);
    
    np_ptr idx = (np_ptr) PyArray_ZEROS(1, (npy_intp *) &n_arr1, NPY_BOOL, 0);
    
    uint n_found = 0;
    npy_double dlon, dlat;
    
    FOR(ii, 0, n_arr1) {
        FOR(jj, 0, n_arr2) {
            dlon = np_delem2(arr1, ii, 0) - np_delem2(arr2, jj, 0);
            dlat = np_delem2(arr1, ii, 1) - np_delem2(arr2, jj, 1);
            
            if ( (dlon * dlon + dlat * dlat) < max_sep) {
                np_belem1(idx, ii) = NPY_TRUE;
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

/**********************
 * Python boilerplate *
 **********************/

init_module(inmet_aux, "inmet_aux",
            pymeth_noargs(test),
            pymeth_varargs(azi_inc),
            pymeth_keywords(asc_dsc_select));
