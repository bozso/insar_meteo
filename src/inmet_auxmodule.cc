#include "capi_structs.hh"
#include "capi_macros.hh"
#include "utils.hh"
#include "satorbit.hh"


typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

typedef nparray<npy_double, 2> array2d;
typedef nparray<npy_double, 1> array1d;
typedef nparray<npy_bool, 1> array1b;


pydoc(test, "test");

static py_ptr test(py_varargs)
{
    array1d arr;
    
    parse_varargs("O", array_type(arr));
    
    if (import(arr))
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
    
    if (import(mean_coords) or import(coeffs) or import(coords))
        return NULL;
    
    npy_intp azi_inc_shape[2] = {(npy_intp) coords.rows(), 2};
    
    if (empty(azi_inc, azi_inc_shape))
        return NULL;
    
    // Set up orbit polynomial structure
    fit_poly orb = {mean_t, start_t, stop_t, mean_coords.data,
                    coeffs.data, is_centered, deg};
    
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
    } // if
    
    return Py_BuildValue("N", get_array(azi_inc));
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
    
    npy_intp idx_shape = (npy_intp) arr1.rows();

    if (empty(idx, &idx_shape))
        return NULL;
    
    max_sep /=  R_earth;
    max_sep = (max_sep * RAD2DEG) * (max_sep * RAD2DEG);
    
    npy_double dlon, dlat;
    uint nfound = 0;
    
    FOR(ii, 0, arr1.rows()) {
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
    
    return Py_BuildValue("NI", get_array(idx), nfound);
} // asc_dsc_select

pydoc(dominant, "dominant");

static py_ptr dominant(py_keywords)
{
    keywords("asc_data", "dsc_data", "cluster_sep");
    
    array2d asc, dsc;
    double max_sep = 100.0;
    
    parse_keywords("OO|d:dominant", array_type(asc), array_type(dsc), &max_sep);
    
    if (import(asc) or import(dsc))
        return NULL;
    
    uint ncluster = 0, nhermite = 0;
    
    array<bool> asc_selected, dsc_selected;
    
    if (init(asc_selected, asc.rows()) or init(dsc_selected, dsc.rows()))
        return NULL;
    
    vector<double> clustered;
    
    
    
    //return Py_BuildValue("NII", clustered.get_array(), ncluster, nhermite);
    Py_RETURN_NONE;
} // dominant


//------------------------------------------------------------------------------

#define version "0.0.1"
#define module_name "inmet_aux"
static const char* module_doc = "inmet_aux";

static PyMethodDef module_methods[] = {
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
        PyErr_SetString(PyExc_ImportError, "can't initialize module"
                        module_name "(failed to import numpy)");
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
