#include "capi_functions.h"
#include "utils.h"
//#include "satorbit.h"

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

#if 0

pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;

    array1d mean_coords;
    array2d coeffs, coords, azi_inc;
    
    parse_varargs("dddIIO!O!O!O!II", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, np_array(mean_coords), np_array(mean_coords),
                  np_array(coords), np_array(azi_inc), &is_lonlat, &max_iter);
    
    
    if (mean_coords.import() or coeffs.import())
        return NULL;
    
    // Set up orbit polynomial structure
    orbit_fit orb(mean_t, start_t, stop_t, mean_coords.get_data(),
                  coeffs.get_data(), is_centered, deg);
    
    if (coords.import() or azi_inc.import())
        return NULL;
    
    uint nrows = coords.rows();
    
    double X, Y, Z, lon, lat, h;
    X = Y = Z = lon = lat = h = 0.0;
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        FOR(ii, 0, nrows) {
            lon = coords(ii, 0) * DEG2RAD;
            lat = coords(ii, 1) * DEG2RAD;
            h   = coords(ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, X, Y, Z);
            
            calc_azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                         azi_inc(ii, 0), azi_inc(ii, 1));
            
        } // for
    }
    // coords contains X, Y, Z
    else {
        FOR(ii, 0, nrows) {
            X = coords(ii, 0);
            Y = coords(ii, 1);
            Z = coords(ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, lon, lat, h);
        
            calc_azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                         azi_inc(ii, 0), azi_inc(ii, 1));
        } // for
    } // else
    
    Py_RETURN_NONE;
} // azi_inc

#endif

init_methods(inmet_aux,
             pymeth_varargs(test))
             //pymeth_varargs(azi_inc))
             
init_module(inmet_aux, "inmet_aux", 0.1)
