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

#include "capi_functions.hpp"
#include "satorbit.hpp"
#include "utils.hpp"

/*****************************************
 * Main functions - calleble from Python *
 *****************************************/

pyfun_doc(test, "test");

static py_ptr test(py_varargs)
{
    py_ptr _array = nullptr;
    np_wrap<npy_double> array;
    
    try {
        logg;
        parse_varargs(args, "O", &_array);
        logg;
        array.import(_array, "arr", 1);
        logg;
    }
    catch(const char * e) {
        errorln("%s", e);
        array.xdecref();
        return NULL;
    }
    
    uint nrows = array.rows();
    
    //array(0) = 1.0;
    //array(1) = 2.0;
    //array(2) = 3.0;
    
    FOR(ii, 0, nrows) {
        prints("%lf ", array(ii));
    }
    
    print("\n");
    
    array.decref();
    Py_RETURN_NONE;
}

#if 0

pyfun_doc(azi_inc, "azi_inc");

static py_ptr azi_inc (py_varargs)
{
    double start_t, stop_t, mean_t;
    uint is_centered, deg, max_iter, is_lonlat;
    py_ptr _coeffs = NULL, _coords = NULL, _mean_coords = NULL;    

    pyfun_parse_varargs("OdddOIIOII:azi_inc", &_coeffs, &start_t, &stop_t,
                        &mean_t, &_mean_coords, &is_centered, &deg, &_coords,
                        &max_iter, &is_lonlat);

    uint n_coords;
    
    // Importing arrays
    np_wrap<npy_double> coeffs, coords, mean_coords, azi_inc;
    
    try {
        coeffs.import(_coeffs, "coeffs", 2);
        coords.import(_coords, "coords", 2);
        mean_coords.import(_mean_coords, "mean_coords", 1);

        /* Coefficients array should be a 2 dimensional 3x(deg + 1) matrix
         * where every row contains the coefficients for the fitted x,y,z
         * polynoms. */

        coeffs.check_matrix(3, deg + 1)
        coords.check_cols(3);
        
        // should be a 3 element vector
        mean_coords.check_rows(3);

        // number of coordinates
        ncoords = coords.rows();
    
        npy_intp azi_inc_shape[2] = {(npy_intp) n_coords, 2};
        
        // matrix holding azimuth and inclination values
        az_inc.empty(2, azi_inc_shape, "azi_inc");
    }
    catch(const char * e) {
        errorln("%s", e);
        coeffs.xdecref();
        coords.xdecref();
        mean_coords.xdecref();
        azi_inc.xdecref();
        return NULL;
    }
    
    // Set up orbit polynomial structure
    orbit_fit orb;
    orb.coeffs = coeffs.get_data();
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.start_t = start_t;
    orb.stop_t = stop_t;
    orb.mean_t = mean_t;
    
    orb.mean_coords = mean_coords.get_data();

    double X, Y, Z, lon, lat, h;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, n_coords) {
            lon = deg2rad(coords(ii, 0));
            lat = deg2rad(coords(ii, 1));
            h   = coords(ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         azi_inc(ii, 0), azi_inc(ii, 1));
            
        }
        // end for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, n_coords) {
            X = coords(ii, 0);
            Y = coords(ii, 1);
            Z = coords(ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, 
                         azi_inc(ii, 0), azi_inc(ii, 1));
        }
        // end for
    }
    // end else

    // Cleanup and return
    coeffs.decref();
    coords.decref();
    mean_coords.decref();
    
    return Py_BuildValue("O", azi_inc.get_array());
} // end azim_inc

pyfun_doc(asc_dsc_select,
"asc_dsc_select");

static py_ptr asc_dsc_select(py_keywords)
{
    char * keywords[] = {"asc", "dsc", "max_sep", NULL};

    py_ptr _arr1 = NULL, _arr2 = NULL;
    npy_double max_sep = 100.0;

    pyfun_parse_keywords(keywords, "OO|d:asc_dsc_select",
                         &_arr1, &_arr2, &max_sep);
    
    npy_double max_sep_deg = max_sep / R_earth;
    max_sep_deg = max_sep_deg * max_sep_deg * RAD2DEG * RAD2DEG;

    println("Maximum separation: %6.3lf meters => approx. %E degrees",
            max_sep, max_sep_deg);

    np_wrap<npy_double> arr1, arr2;
    np_wrap<npy_bool> idx;
    
    uint n_arr1, n_arr2;
    
    try {
        arr1.import(_arr1, "arr1", 2);
        arr2.import(_arr2, "arr2", 2);
    
        n_arr1 = arr1.rows();
        n_arr2 = arr2.rows();
        
        idx.empty(1, (npy_intp *) &n_arr1, 0, "idx");
    }
    catch(const char * e) {
        errorln("%s", e);
        arr1.xdecref();
        arr2.xdecref();
        idx.xdecref();
        return NULL;
    }
    
    uint n_found = 0;
    npy_double dlon, dlat;
    
    FOR(ii, 0, n_arr1) {
        FOR(jj, 0, n_arr2) {
            dlon = arr1(ii, 0) - arr2(jj, 0);
            dlat = arr1(ii, 1) - arr2(jj, 1);
            
            if ( (dlon * dlon + dlat * dlat) < max_sep) {
                idx(ii) = NPY_TRUE;
                n_found++;
                break;
            }
        }
    }
    
    arr1.decref();
    arr2.decref();
    return Py_BuildValue("OI", idx.np_array(), n_found);
}
// end asc_dsc_select

#endif

// Module initialization

init_table(inmet_aux, pymeth_varargs(test));

init_module(inmet_aux, "inmet_aux")
