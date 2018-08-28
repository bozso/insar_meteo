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

#include "capi_functions.h"
#include "satorbit.h"
#include "utils.h"

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

pyfun_doc(test, "test");

static py_ptr test(py_ptr self, py_ptr args)
{
    py_ptr _array;
    parse_varargs(args, "O", &_array);
    
    ar_double array;
    ar_import_check_double(array, _array, 1, "array");
    
    uint nrows = ar_rows(array);
    
    FOR(ii, 0, nrows)
        printf("%lf ", ar_elem1(array, ii));
    printf("\n");

    ar_decref(array);
    Py_RETURN_NONE;
fail:
    ar_xdecref(array);
    return NULL;
}

pyfun_doc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_ptr self, py_ptr args)
{
    double start_t, stop_t, mean_t;
    uint is_centered, deg, max_iter, is_lonlat;
    py_ptr _coeffs = NULL, _coords = NULL, _mean_coords = NULL;    

    parse_varargs(args, "OdddOIIOII:azi_inc", &_coeffs, &start_t, &stop_t,
                  &mean_t, &_mean_coords, &is_centered, &deg, &_coords,
                  &max_iter, &is_lonlat);

    ar_double coeffs, coords, mean_coords, azi_inc;
    
    ar_import_check_double(coeffs, _coeffs, 2, "coeffs");
    ar_import_check_double(coords, _coords, 2, "coords");
    ar_import_check_double(mean_coords, _mean_coords, 1, "mean_coords");
    
    /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix where
     * every row contains the coefficients for the fitted x,y,z polynoms. */
    ar_check_matrix(coeffs, 3, deg + 1);
    
    // should be nx3 matrix
    ar_check_cols(coords, 3);

    // should be a 3 element vector
    ar_check_rows(mean_coords, 3);
    
    // number of coordinates
    uint n_coords = ar_rows(coords);

    npy_intp azi_inc_shape[2] = {(npy_intp) n_coords, 2};
    
    // matrix holding azimuth and inclination values
    ar_empty_double(azi_inc, 2, azi_inc_shape, "azi_inc");
    
    // Set up orbit polynomial structure
    orbit_fit orb;
    orb.coeffs = ar_data(coeffs);
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.start_t = start_t;
    orb.stop_t = stop_t;
    orb.mean_t = mean_t;
    
    orb.mean_coords = ar_data(mean_coords);

    npy_double X, Y, Z, lon, lat, h;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, n_coords) {
            lon = ar_elem2(coords, ii, 0) * DEG2RAD;
            lat = ar_elem2(coords, ii, 1) * DEG2RAD;
            h   = ar_elem2(coords, ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         ar_ptr2(azi_inc, ii, 0),
                         ar_ptr2(azi_inc, ii, 1));
            
        }
        // end for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, n_coords) {
            X = ar_elem2(coords, ii, 0);
            Y = ar_elem2(coords, ii, 1);
            Z = ar_elem2(coords, ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         ar_ptr2(azi_inc, ii, 0),
                         ar_ptr2(azi_inc, ii, 1));
        }
        // end for
    }
    // end else

    // Cleanup and return
    ar_decref(coeffs);
    ar_decref(coords);
    ar_decref(mean_coords);
    return Py_BuildValue("O", azi_inc);

fail:
    ar_xdecref(coeffs);
    ar_xdecref(coords);
    ar_xdecref(mean_coords);
    ar_xdecref(azi_inc);
    return NULL;
} // end azim_inc

pyfun_doc(asc_dsc_select, "asc_dsc_select");

static py_ptr asc_dsc_select(py_ptr self, py_ptr args, py_ptr kwargs)
{
    char * keywords[] = {"asc", "dsc", "max_sep", NULL};

    py_ptr _arr1 = NULL, _arr2 = NULL;
    double max_sep = 100.0;

    parse_keywords(args, kwargs, keywords, "OO|d:asc_dsc_select",
                   &_arr1, &_arr2, &max_sep);
    
    double max_sep_deg = max_sep / R_earth;
    max_sep_deg = max_sep_deg * max_sep_deg * RAD2DEG * RAD2DEG;

    println("Maximum separation: %6.3lf meters => approx. %E degrees",
             max_sep, max_sep_deg);

    ar_double arr1, arr2;
    ar_import_check_double(arr1, _arr1, 2, "asc");
    ar_import_check_double(arr2, _arr2, 2, "dsc");
    
    uint n_arr1 = ar_rows(arr1), n_arr2 = ar_rows(arr2);
    
    ar_bool idx;
    ar_empty_bool(idx, 1, (npy_intp *) &n_arr1, "idx");
    
    uint n_found = 0;
    npy_double dlon, dlat;
    
    FOR(ii, 0, n_arr1) {
        FOR(jj, 0, n_arr2) {
            dlon = ar_elem2(arr1, ii, 0) - ar_elem2(arr2, jj, 0);
            dlat = ar_elem2(arr1, ii, 1) - ar_elem2(arr2, jj, 1);
            
            if ( (dlon * dlon + dlat * dlat) < max_sep) {
                ar_elem1(idx, ii) = NPY_TRUE;
                n_found++;
                break;
            }
        }
    }
    
    ar_decref(arr1);
    ar_decref(arr2);
    return Py_BuildValue("OI", idx, n_found);

fail:
    ar_xdecref(arr1);
    ar_xdecref(arr2);
    return NULL;
}
// end asc_dsc_select


/**********************
 * Python boilerplate *
 **********************/

init_table(inmet_aux,
           pymeth_varargs(test),
           pymeth_varargs(azi_inc),
           pymeth_keywords(asc_dsc_select)
           );

init_module(inmet_aux, "inmet_aux")
            
