#include "capi_functions.hpp"
#include "utils.hpp"
//#include "satorbit.hpp"

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

typedef array<npy_double, 3> array3d;
typedef array<npy_double, 2> array2d;
typedef array<npy_double, 1> array1d;

typedef array<npy_int, 3> array3i;
typedef array<npy_int, 2> array2i;
typedef array<npy_int, 1> array1i;

typedef arraynd<npy_double> arrayd;
typedef arraynd<npy_int> arrayi;

pydoc(test, "test");

static py_ptr test(py_varargs)
{
    array1d arr;
    
    parse_varargs("O!", np_array(arr));
    
    if (arr.import())
        return NULL;
    
    FOR(ii, 0, arr.rows())
        print("%lf ", arr(ii));

    prints("\n");

    Py_RETURN_NONE;
}

#if 0

pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;

    ar_double mean_coords, coeffs, coords, azi_inc;
    
    parse_varargs("dddIIO!O!O!O!II", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, np_array(mean_coords), np_array(mean_coords),
                  np_array(coords), np_array(azi_inc), &is_lonlat, &max_iter);
    
    if (ar_import_check(mean_coords, 1) or ar_import_check(coeffs, 2) or
        ar_import_check(coords, 2) or ar_import_check(azi_inc, 2))
        goto fail;
    
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
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter,
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
        
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter,
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

#endif 

init_methods(inmet_aux,
             pymeth_varargs(test))
             //pymeth_varargs(azi_inc))
             
init_module(inmet_aux, "inmet_aux", 0.1)
