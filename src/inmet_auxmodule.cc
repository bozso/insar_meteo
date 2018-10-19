#include "pymacros.hh"
#include "nparray.hh"
#include "array.hh"
#include "satorbit.hh"
#include "utils.hh"


typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

typedef nparray<npy_double, 2> array2d;
typedef nparray<npy_double, 1> array1d;
typedef nparray<npy_bool, 1> array1b;


pydoc(ell_to_merc_full, "ell_to_merc_full");

static py_ptr ell_to_merc_full(py_varargs)
{
    array1d lon, lat;
    double a, e, lon0;
    uint isdeg;
    
    parse_varargs("OOdddI", array_type(lon), array_type(lat), &lon0,
                  &a, &e, &isdeg);
    
    if (lon.import() or lat.import())
        return NULL;
    
    size_t rows = lon.rows();
    
    if (rows != lat.rows()) {
        // TODO: set error message !!!
        return NULL;
    }
    
    array2d xy;
    npy_intp shape[2] = {npy_intp(rows), 2};
    
    if (xy.empty(shape))
        return NULL;
    
    if (isdeg) {
        FOR(ii, 0, rows) {
            xy(ii,0) = a * deg2rad * (lon(ii) - lon0);
            
            double sin_lat = sin(deg2rad * lat(ii));
            double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
            
            xy(ii,1) = a * (tan((pi_per_4 + lat(ii) / 2.0)) * tmp);
        }
    } else {
        FOR(ii, 0, rows) {
            xy(ii,0) = a * (lon(ii) - lon0);
            
            double sin_lat = sin(lat(ii));
            double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
            
            xy(ii,1) = a * (tan((pi_per_4 + lat(ii) / 2.0)) * tmp);
        }
    }
    
    return Py_BuildValue("N", ret(xy));
}


pydoc(ell_to_merc_fast, "ell_to_merc_fast");

static py_ptr ell_to_merc_fast(py_varargs)
{
    array1d lon, lat;
    double a, e, lon0;
    uint isdeg;
    
    parse_varargs("OOdddI", array_type(lon), array_type(lat), &lon0,
                  &a, &e, &isdeg);
    
    if (lon.import() or lat.import())
        return NULL;
    
    size_t rows = lon.rows();
    
    if (rows != lat.rows()) {
        // TODO: set error message !!!
        return NULL;
    }
    
    array2d xy;
    npy_intp shape[2] = {npy_intp(rows), 2};
    
    if (xy.empty(shape))
        return NULL;

    
    if (isdeg) {
        FOR(ii, 0, rows) {
            xy(ii,0) = a * deg2rad * (lon(ii) - lon0);
            
            double scale = xy(ii,0) / lon(ii);
            
            xy(ii,1) = rad2deg * log(tan(pi_per_4 + lat(ii) * deg2rad / 2.0)) * scale;
        }
    } else {
        FOR(ii, 0, rows) {
            xy(ii,0) = a * (lon(ii) - lon0);
            
            double scale = xy(ii,0) / lon(ii);
            
            xy(ii,1) = rad2deg * log(tan(pi_per_4 + lat(ii) / 2.0)) * scale;
        }
    }
    
    return Py_BuildValue("N", ret(xy));
}

pydoc(test, "test");

static py_ptr test(py_varargs)
{
    array1d arr;
    
    parse_varargs("O", array_type(arr));
    
    if (arr.import())
        return NULL;
    
    FOR(ii, 0, arr.rows())
        printf("%lf ", arr(ii));

    printf("\n");

    Py_RETURN_NONE;
}


pydoc(azi_inc, "azi_inc");

static py_ptr azi_inc(py_varargs)
{
    double mean_t = 0.0, start_t = 0.0, stop_t = 0.0;
    uint is_centered = 0, deg = 0, is_lonlat = 0, max_iter = 0;
    
    array1d mean_coords;
    array2d coeffs, coords, azi_inc;
    
    parse_varargs("dddIIOOOII", &mean_t, &start_t, &stop_t, &is_centered,
                  &deg, array_type(mean_coords), array_type(mean_coords),
                  array_type(coords), &is_lonlat, &max_iter);
    
    if (mean_coords.import() or coeffs.import() or coords.import())
        return NULL;
    
    npy_intp azi_inc_shape[2] = {(npy_intp) coords.rows(), 2};
    
    if (azi_inc.empty(azi_inc_shape))
        return NULL;
    
    // Set up orbit polynomial structure
    fit_poly orb(mean_t, start_t, stop_t, mean_coords.data, coeffs,
                 is_centered, deg);
    
    calc_azi_inc(orb, coords, azi_inc, max_iter, is_lonlat);
    
    return Py_BuildValue("N", ret(azi_inc));
} // azi_inc

pydoc(asc_dsc_select, "asc_dsc_select");

static py_ptr asc_dsc_select(py_keywords)
{
    keywords("array1", "array2", "max_sep");
    
    array2d arr1, arr2;
    array1b idx;
    double max_sep = 100.0;
    
    parse_keywords("OO|d:asc_dsc_select", array_type(arr1), array_type(arr2),
                                          &max_sep);
    
    size_t rows = arr1.rows();
    
    npy_intp idx_shape = (npy_intp) rows;

    if (idx.zeros(&idx_shape))
        return NULL;
    
    max_sep /=  R_earth;
    max_sep = (max_sep * rad2deg) * (max_sep * rad2deg);
    
    npy_double dlon, dlat;
    uint nfound = 0;
    
    FOR(ii, 0, rows) {
        FOR(jj, 0, arr2.rows()) {
            dlon = arr1(ii,0) - arr2(jj,0);
            dlat = arr1(ii,1) - arr2(jj,1);
            
            if ((dlon * dlon + dlat * dlat) < max_sep) {
                idx(ii) = NPY_TRUE;
                nfound++;
                break;
            }
        }
    }
    
    return Py_BuildValue("NI", ret(idx), nfound);
} // asc_dsc_select

pydoc(dominant, "dominant");

static py_ptr dominant(py_keywords)
{
    keywords("asc_data", "dsc_data", "cluster_sep");
    
    array2d asc, dsc;
    double max_sep = 100.0;
    
    parse_keywords("OO|d:dominant", array_type(asc), array_type(dsc), &max_sep);
    
    if (asc.import() or dsc.import())
        return NULL;
    
    uint ncluster = 0, nhermite = 0;
    
    array<bool> asc_selected, dsc_selected;
    
    if (asc_selected.init(asc.rows()) or dsc_selected.init(dsc.rows()))
        return NULL;
    
    //vector<double> clustered;
    
    //return Py_BuildValue("NII", ret(clustered), ncluster, nhermite);
    Py_RETURN_NONE;
} // dominant


//------------------------------------------------------------------------------

#define version "0.0.1"
#define module_name "inmet_aux"
static const char* module_doc = "inmet_aux";

static PyMethodDef module_methods[] = {
    pymeth_varargs(ell_to_merc_fast),
    pymeth_varargs(ell_to_merc_full),
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
