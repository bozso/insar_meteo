#include "pymacros.hh"
#include "nparray.hh"
#include "view.hh"
#include "array.hh"
#include "satorbit.hh"
#include "utils.hh"


typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;


pydoc(ell_to_merc, "ell_to_merc");

static py_ptr ell_to_merc(py_varargs)
{
    py_ptr plon(NULL), plat(NULL);
    double a, e, lon0;
    uint isdeg, fast;

    parse_varargs("OOdddII", &plon, &plat, &lon0, &a, &e, &isdeg, &fast);
    
    nparray _lon(dt_double, 1, plon), _lat(dt_double, 1, plat); check_error;
    
    size_t rows = _lon.shape[0];
    
    if (_lat.check_rows(rows))
        return NULL;
    
    nparray _xy(dt_double, empty, 'c', npy_intp(rows), 2);
    
    view<double> lon(_lon), lat(_lat), xy(_xy);
    
    if (isdeg) {
        if (fast) {
            FOR(ii, rows) {
                xy(ii,0) = a * deg2rad * (lon(ii) - lon0);
                
                double scale = xy(ii,0) / lon(ii);
                
                xy(ii,1) = rad2deg * log(tan(pi_per_4 + lat(ii) * deg2rad / 2.0)) * scale;
            }
        } else {
            FOR(ii, rows) {
                xy(ii,0) = a * deg2rad * (lon(ii) - lon0);
                
                double sin_lat = sin(deg2rad * lat(ii));
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
                
                xy(ii,1) = a * (tan((pi_per_4 + lat(ii) / 2.0)) * tmp);
            }
        }
    } else {
        if (fast) {
            FOR(ii, rows) {
                xy(ii,0) = a * (lon(ii) - lon0);
                
                double scale = xy(ii,0) / lon(ii);
                
                xy(ii,1) = rad2deg * log(tan(pi_per_4 + lat(ii) / 2.0)) * scale;
            }
        } else {
            FOR(ii, rows) {
                xy(ii,0) = a * (lon(ii) - lon0);
                
                double sin_lat = sin(lat(ii));
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
                
                xy(ii,1) = a * (tan((pi_per_4 + lat(ii) / 2.0)) * tmp);
            }
        }
    }
    
    return Py_BuildValue("N", _xy.ret());
}


pydoc(test, "test");

static py_ptr test(py_varargs)
{
    py_ptr parr(NULL);
    parse_varargs("O", &parr);
    
    nparray _arr(dt_double, 1, parr); check_error;
    
    view<npy_double> arr(_arr);
    
    FORZ(ii, arr.shape[0])
        printf("%lf ", arr(ii));

    printf("\n");

    Py_RETURN_NONE;
}


pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    py_ptr pmean_coords(NULL), pcoeffs(NULL), pcoords(NULL);
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;
    
    parse_varargs("dddIIOOOII", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, &pmean_coords, &pmean_coords, &pcoords,
                  &is_lonlat, &max_iter);
    
    nparray _mean_coords(dt_double, 1, pmean_coords),
            _coeffs(dt_double, 2, pcoeffs), _coords(dt_double, 2, pcoords),
            _azi_inc(dt_double, empty, 'c', npy_intp(_coords.shape[0]), 2);
    check_error;
    
    view<npy_double> coeffs(_coeffs), coords(_coords), azi_inc(_azi_inc);
    
    // Set up orbit polynomial structure
    fit_poly orb(mean_t, start_t, stop_t, _mean_coords.data<npy_double>(),
                 coeffs, is_centered, deg);
    
    calc_azi_inc(orb, coords, azi_inc, max_iter, is_lonlat);
    
    return Py_BuildValue("N", _azi_inc.ret());
} // azi_inc


pydoc(asc_dsc_select, "asc_dsc_select");

static py_ptr asc_dsc_select(py_keywords)
{
    keywords("array1", "array2", "max_sep");
    
    py_ptr parr1(NULL), parr2(NULL);
    double max_sep = 100.0;
    
    parse_keywords("OO|d:asc_dsc_select", &parr1, &parr2, &max_sep);
    
    nparray _arr1(dt_double, 2, parr1), _arr2(dt_double, 2, parr2),
            _idx(dt_double, zeros, 'c', _arr1.shape[0]); check_error;
    
    max_sep /=  R_earth;
    max_sep = (max_sep * rad2deg) * (max_sep * rad2deg);
    
    npy_double dlon, dlat;
    uint nfound = 0;
    
    view<npy_double> arr1(_arr1), arr2(_arr2);
    view<npy_bool> idx(_idx);
    
    FOR(ii, arr1.shape[0]) {
        FOR(jj, arr2.shape[0]) {
            dlon = arr1(ii,0) - arr2(jj,0);
            dlat = arr1(ii,1) - arr2(jj,1);
            
            if ((dlon * dlon + dlat * dlat) < max_sep) {
                idx(ii) = NPY_TRUE;
                nfound++;
                break;
            }
        }
    }
    
    return Py_BuildValue("NI", _idx.ret(), nfound);
} // asc_dsc_select


pydoc(dominant, "dominant");

static py_ptr dominant(py_keywords)
{
    keywords("asc_data", "dsc_data", "cluster_sep");
    
    py_ptr pasc(NULL), pdsc(NULL);
    double max_sep = 100.0;
    parse_keywords("OO|d:dominant", &pasc, &pdsc, &max_sep);
    
    nparray _asc(dt_double, 2, pasc), _dsc(dt_double, 2, pdsc); check_error;
    
    uint ncluster = 0, nhermite = 0;
    
    //array<bool> asc_selected(_asc.shape[0]), dsc_selected(_dsc.shape[0]);
    //check_error;
    
    //dmat<double> clustered;
    
    //return Py_BuildValue("NII", ret(clustered), ncluster, nhermite);
    Py_RETURN_NONE;
} // dominant


//------------------------------------------------------------------------------

#define version "0.0.1"
#define module_name "inmet_aux"
static char const* module_doc = "inmet_aux";

static PyMethodDef module_methods[] = {
    pymeth_varargs(ell_to_merc),
    pymeth_varargs(test),
    pymeth_varargs(azi_inc),
    pymeth_keywords(asc_dsc_select),
    pymeth_keywords(dominant),
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
