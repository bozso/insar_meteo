#include "pymacros.hh"
#include "Python.h"
#include "numpy/arrayobject.h"


typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

pydoc(test, "test");

static py_ptr test(py_varargs)
{
    py_ptr parr(NULL);
    parse_varargs("O", &parr);
    
    np_ptr aa = (np_ptr) PyArray_FROM_OTF(parr, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (aa == NULL)
        return NULL;
    
    size_t n = PyArray_SHAPE(aa)[0];
    
    double sum = 0.0;
    
    for(size_t ii = n; ii--;)
        sum += *(double*) PyArray_GETPTR1(aa, ii) + 1.0;
    
    //printf("Sum: %lf\n", sum);

    Py_CLEAR(aa);
    Py_RETURN_NONE;
}


//------------------------------------------------------------------------------

#define version "0.0.1"
#define module_name "inmet_aux"
static char const* module_doc = "inmet_aux";

static PyMethodDef module_methods[] = {
    pymeth_varargs(test),
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
PyMODINIT_FUNC CONCAT(PyInit_, cext)(void) {
#else
#define RETVAL
PyMODINIT_FUNC CONCAT(init, cext)(void) {
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
