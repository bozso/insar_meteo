
#include "capi_functions.hpp"
#include "utils.hpp"
#include "satorbit.hpp"

pydoc(test, "test");

static py_ptr test(py_varargs)
{
    array2d arr;
    np_ptr _arr = NULL;
    
    parse_varargs("O!", &np_type, &_arr);
    
    if(arr.import(_arr))
        return NULL;
    
    FOR(ii, 0, arr.rows()) {
        FOR(jj, 0, arr.cols())
            print("%lf ", arr(ii,jj));
        prints("\n");
    }
    
    Py_RETURN_NONE;
}

pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;
    np_ptr _coords = NULL, _azi_inc = NULL, _mean_coords = NULL, _coeffs = NULL;
    
    parse_varargs("dddIIO!O!O!O!II", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, np_array(_mean_coords), np_array(_mean_coords),
                  np_array(_coords), np_array(_azi_inc), &is_lonlat, &max_iter);
    
    array1d mean_coords;
    array2d coeffs;
    
    if (mean_coords.import(_mean_coords) or coeffs.import(_coeffs))
        return NULL;
    
    // Set up orbit polynomial structure
    orbit_fit orb(mean_t, start_t, stop_t, mean_coords.get_data(),
                  coeffs.get_data(), is_centered, deg);
    
    array2d coords, azi_inc;
    
    if (coords.import(_coords) or azi_inc.import(_azi_inc))
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


init_methods(inmet_aux,
             pymeth_varargs(test),
             pymeth_varargs(azi_inc))
             
init_module(inmet_aux, "inmet_aux", 0.1)
