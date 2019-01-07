#include <stdio.h>

#include "nparray.h"
#include "math_aux.h"
#include "pymacros.h"
#include "satorbit.h"
#include "utils.h"
#include "common.h"
#include "view.h"


typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;


static py_ptr ell_to_merc(py_varargs)
{
    py_ptr plon, plat;
    plon = plat = NULL;
    
    double a, e, lon0;
    uint isdeg, fast;

    parse_varargs("OOdddII", &plon, &plat, &lon0, &a, &e, &isdeg, &fast);
    
    nparray _lon = NULL, _lat = NULL, _xy = NULL;
    
    if (from_otf(&_lon, np_cdouble, 1, plon) or
        from_otf(&_lat, np_cdouble, 1, plat))
        goto fail;

    size_t rows = _lon->shape[0];

    if (check_rows(_lat, rows))
        goto fail;
    
    if (newarray(&_xy, np_cdouble, empty, 'c', (npy_intp) rows, (npy_intp[]){2}))
        goto fail;
    
    view_double lon, lat, xy;
    setup_view(lon, _lon); setup_view(lat, _lat); setup_view(xy, _xy);
    
    
    if (isdeg) {
        if (fast) {
            m_forz(ii, rows) {
                double Lon = ar_elem1(lon, ii);
                double Lat = ar_elem1(lat, ii);
                double tmp = a * deg2rad * (Lon - lon0);
                
                ar_elem2(xy, ii, 0) = tmp;
                
                ar_elem2(xy, ii, 1) =
                rad2deg * log(tan(pi_per_4 + Lat * deg2rad / 2.0)) * (tmp / Lon);
            }
        } else {
            m_forz(ii, rows) {
                double Lat = ar_elem1(lat, ii);
                ar_elem2(xy, ii, 0) = a * deg2rad * (ar_elem1(lon, ii) - lon0);
                
                double sin_lat = sin(deg2rad * Lat);
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat), e / 2.0);
                
                ar_elem2(xy, ii, 1) = a * (tan((pi_per_4 + Lat / 2.0)) * tmp);
            }
        }
    } else {
        if (fast) {
            m_forz(ii, rows) {
                double Lon = ar_elem1(lon, ii);
                double Lat = ar_elem1(lat, ii);
                double tmp = a * (Lon - lon0);
                
                ar_elem2(xy, ii, 0) = tmp;
                
                ar_elem2(xy, ii, 1) =
                rad2deg * log(tan(pi_per_4 + Lat * deg2rad / 2.0)) * (tmp / Lon);
            }
        } else {
            m_forz(ii, rows) {
                double Lat = ar_elem1(lat, ii);
                ar_elem2(xy, ii, 0) = a * (ar_elem1(lon, ii) - lon0);
                
                double sin_lat = sin(Lat);
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
                
                ar_elem2(xy, ii, 1) = a * (tan((pi_per_4 + Lat / 2.0)) * tmp);
            }
        }
    }
    
    del(_lon); del(_lat);
    return Py_BuildValue("N", _xy->npobj);

fail:
    del(_lon); del(_lat); del(_xy);
    return NULL;
}


static py_ptr azi_inc(py_varargs)
{
    py_ptr pmean_coords = NULL, pcoeffs = NULL, pcoords = NULL;
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;
    
    parse_varargs("dddIIOOOII", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, &pmean_coords, &pmean_coords, &pcoords,
                  &is_lonlat, &max_iter);
    
    nparray _mean_coords = NULL, _coeffs = NULL, _coords = NULL, _azi_inc = NULL;
    
    if (from_otf(&_mean_coords, np_cdouble, 1, pmean_coords) or
        from_otf(&_coeffs, np_cdouble, 2, pcoeffs) or
        from_otf(&_coords, np_cdouble, 2, pcoords) or
        newarray(&_azi_inc, np_cdouble, empty, 'c',
                 _coords->shape[0], (npy_intp[]){2}))
        goto fail;
    
    view_double coeffs;
    setup_view(coeffs, _coeffs);
    
    
    // Set up orbit polynomial structure
    fit_poly orb = {mean_t, start_t, stop_t, ar_data(_mean_coords),
                    &coeffs, is_centered, deg};
    
    calc_azi_inc(&orb, _coords, _azi_inc, max_iter, is_lonlat);
    
    del(_mean_coords); del(_coeffs); del(_coords);
    return Py_BuildValue("N", _azi_inc->npobj);

fail:
    del(_mean_coords); del(_coeffs); del(_coords); del(_azi_inc);
    return NULL;
} // azi_inc


static py_ptr test(py_varargs)
{
    py_ptr parr = NULL;
    parse_varargs("O", &parr);
    
    nparray _arr = NULL;
    
    m_log;
    if (from_otf(&_arr, np_double, 1, parr))
        return NULL;
    

    view_double arr;
    setup_view(arr, _arr);
    
    size_t n = _arr->shape[0];
    
    double sum = 0.0;
    
    m_forz(ii, n)
        sum += ar_elem1(arr, ii) + 1.0;
    
    printf("Sum: %lf\n", sum);
    
    del(_arr);
    Py_RETURN_NONE;
}


static py_ptr asc_dsc_select(py_keywords)
{
    keywords("array1", "array2", "max_sep");
    
    py_ptr parr1 = NULL, parr2 = NULL;
    double max_sep = 100.0;
    
    parse_keywords("OO|d:asc_dsc_select", &parr1, &parr2, &max_sep);
    
    nparray _arr1 = NULL, _arr2 = NULL, _idx = NULL;
    
    if (from_otf(&_arr1, np_double, 2, parr1) or
        from_otf(&_arr2, np_double, 2, parr2) or
        newarray(&_idx, np_double, zeros, 'c', 1,
                            (npy_intp[]){(npy_intp)_arr1->shape[0]}))
        goto fail;

            
    max_sep /=  R_earth;
    max_sep = (max_sep * rad2deg) * (max_sep * rad2deg);
    
    uint nfound = 0;
    size_t n1 = _arr1->shape[0], n2 = _arr2->shape[0];
    
    view_double arr1, arr2;
    setup_view(arr1, _arr1); setup_view(arr2, _arr2);
    view_bool idx;
    setup_view(idx, _idx);
    
    m_forz(ii, n1) {
        m_forz(jj, n2) {
            double dlon = ar_elem2(arr1, ii, 0) - ar_elem2(arr2, jj, 0),
                   dlat = ar_elem2(arr1, ii, 1) - ar_elem2(arr2, jj, 1);
            
            if ((dlon * dlon + dlat * dlat) < max_sep) {
                *(ar_ptr1(idx, ii)) = NPY_TRUE;
                nfound++;
                break;
            }
        }
    }
    
    del(_arr1); del(_arr2);
    return Py_BuildValue("NI", _idx->npobj, nfound);

fail:
    del(_arr1); del(_arr2); del(_idx);
    return NULL;
} // asc_dsc_select


#if 0


static py_ptr dominant(py_keywords)
{
    keywords("asc_data", "dsc_data", "cluster_sep");
    
    py_ptr pasc(NULL), pdsc(NULL);
    double max_sep = 100.0;
    parse_keywords("OO|d:dominant", &pasc, &pdsc, &max_sep);
    
    nparray _asc(np_cdouble, 2, pasc), _dsc(np_cdouble, 2, pdsc);
    err_check(NULL);
    
    uint ncluster = 0, nhermite = 0;
    
    //array<bool> asc_selected(_asc.shape[0]), dsc_selected(_dsc.shape[0]);
    //check_error;
    
    //dmat<double> clustered;
    
    //return Py_BuildValue("NII", ret(clustered), ncluster, nhermite);
    Py_RETURN_NONE;
} // dominant

#endif

//------------------------------------------------------------------------------

#define version "0.0.1"
#define module_name "inmet_aux"
static char const* module_doc = "inmet_aux";

static PyMethodDef module_methods[] = {
    pymeth_varargs(ell_to_merc, "ell_to_merc"),
    pymeth_varargs(test, "test"),
    pymeth_varargs(azi_inc, "azi_inc"),
    pymeth_keywords(asc_dsc_select, "asc_dsc_select"),
    //pymeth_keywords(dominant, "dominant"),
    {NULL, NULL, 0, NULL}
};

//------------------------------------------------------------------------------

static PyObject * module_error;
static PyObject * module;

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    module_name,
    NULL,
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyMODINIT_FUNC CONCAT(PyInit_, inmet_aux)(void) {
#else
#define RETVAL
PyMODINIT_FUNC CONCAT(init, inmet_aux)(void) {
#endif
    PyObject *m, *d, *s;
#if PY_VERSION_HEX >= 0x03000000
    m = module = PyModule_Create(&module_def);
#else
    m = module = Py_InitModule(module_name, module_methods);
#endif
    import_array();
    
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_ImportError, "Cannot initialize module "
                        module_name " (failed to import numpy)");
        return RETVAL;
    }
    d = PyModule_GetDict(m);
    s = PyString_FromString("$Revision: $");
    
    PyDict_SetItemString(d, "__version__", s);

#if PY_VERSION_HEX >= 0x03000000
    s = PyUnicode_FromString(
#else
    s = PyString_FromString(
#endif
    module_doc);
  
    PyDict_SetItemString(d, "__doc__", s);

    module_error = PyErr_NewException(module_name ".error", NULL, NULL);

    Py_DECREF(s);

    return RETVAL;
}
