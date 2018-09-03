#include <iso646.h>
#include "capi_functions.h"
#include "utils.h"
#include "satorbit.h"

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

pydoc(test, "test");

static py_ptr test(py_varargs)
{
    ar_double array;
    
    parse_varargs("O!", np_array(array));
    
    ar_import(array, 1);
    
    FOR(ii, 0, ar_rows(array))
        print("%lf ", ar_elem1(array, ii));

    prints("\n");

    ar_free(array);
    Py_RETURN_NONE;

fail:
    ar_xfree(array);
    return NULL;
}

pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;

    ar_double mean_coords, coeffs, coords, azi_inc;
    
    parse_varargs("dddIIO!O!O!O!II", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, np_array(mean_coords), np_array(mean_coords),
                  np_array(coords), np_array(azi_inc), &is_lonlat, &max_iter);
    
    ar_import(mean_coords, 1);
    ar_import(coeffs, 2);
    ar_import(coords, 2);
    ar_import(azi_inc, 2);
    
    // Set up orbit polynomial structure
    orbit_fit orb = {mean_t, start_t, stop_t, mean_coords.data,
                     coeffs.data, is_centered, deg};
    
    uint nrows = ar_rows(coords);
    
    double X, Y, Z, lon, lat, h;
    X = Y = Z = lon = lat = h = 0.0;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, nrows) {
            lon = ar_elem2(coords, ii, 0) * DEG2RAD;
            lat = ar_elem2(coords, ii, 1) * DEG2RAD;
            h   = ar_elem2(coords, ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, h, max_iter,
                         ar_ptr2(azi_inc, ii, 0), ar_ptr2(azi_inc, ii, 1));
            
        } // for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, nrows) {
            X = ar_elem2(coords, ii, 0);
            Y = ar_elem2(coords, ii, 1);
            Z = ar_elem2(coords, ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, &lon, &lat, &h);
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, h, max_iter,
                         ar_ptr2(azi_inc, ii, 0), ar_ptr2(azi_inc, ii, 1));
        } // for
    } // else
    
    ar_free(mean_coords);
    ar_free(coeffs);
    ar_free(coords);
    ar_free(azi_inc);
    Py_RETURN_NONE;

fail:
    ar_xfree(mean_coords);
    ar_xfree(coeffs);
    ar_xfree(coords);
    ar_xfree(azi_inc);
    return NULL;
} // azi_inc

init_methods(inmet_aux,
             pymeth_varargs(test),
             pymeth_varargs(azi_inc))
             
init_module(inmet_aux, "inmet_aux", 0.1)
