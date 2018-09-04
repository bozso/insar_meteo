#ifndef CAPI_FUNCTIONS_H
#define CAPI_FUNCTIONS_H

#define CONCAT(a,b) a ## b
#define QUOTE(a) #a

#include "Python.h"
#include "numpy/arrayobject.h"

#define pyexc(exc_type, format, ...)\
        PyErr_Format((exc_type), (format), __VA_ARGS__)\

#define pyexcs(exc_type, string)\
        PyErr_Format((exc_type), (string))\

/*************
 * IO macros *
 *************/

#define _log PySys_WriteStderr("FILE: %s :: LINE: %d\n", __FILE__, __LINE__)

#define errors(text) PySys_WriteStderr(text)
#define error(string, ...) PySys_WriteStdout(string, __VA_ARGS__)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define prints(string) PySys_WriteStdout(string)
#define print(string, ...) PySys_WriteStdout(string, __VA_ARGS__)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)


/************************************
 * Wrapper object for numpy arrays. *
 ************************************/

typedef struct array_descr_t {
    unsigned int ndim, *strides;
    npy_intp *shape;
    PyArrayObject * np_array;
} array_descr;

typedef struct ar_double_t {
    npy_double * data;
    array_descr _array;
} ar_double;

typedef struct ar_bool_t {
    npy_bool * data;
    array_descr _array;
} ar_bool;


static int _ar_import_from_np(ar_dsc, data, edim, _array)
    array_descr * ar_dsc;
    void **data;
    unsigned int edim;
    PyArrayObject * _array;
{
    unsigned int ii = 0;

    ar_dsc->shape = NULL;
    ar_dsc->strides = NULL;

    unsigned int _ndim = (unsigned int) PyArray_NDIM(_array);
    
    if (_ndim != edim) {
        pyexc(PyExc_TypeError, "numpy array expected to be %u "
                               "dimensional but we got %u dimensional "
                               "array!", edim, _ndim);

        return 1;
    }
    
    ar_dsc->ndim = _ndim;
    
    ar_dsc->shape = PyArray_DIMS(_array);
    
    int elemsize = (int) PyArray_ITEMSIZE(_array);
    
    if ((ar_dsc->strides = PyMem_New(unsigned int, _ndim)) == NULL) {
        pyexcs(PyExc_MemoryError, "Failed to allocate memory for array "
                                  "strides!");
        return 1;
    }
    
    npy_intp * _strides = PyArray_STRIDES(_array);
    
    for(ii = 0; ii < _ndim; ++ii)
        ar_dsc->strides[ii] = (unsigned int) ((double) _strides[ii] / elemsize);
    
    ar_dsc->np_array = _array;
    
    *data = PyArray_DATA(_array);
    
    return 0;
}

#define ar_import_from_np(ar_struct, np_ptr, edim)\
        _ar_import_from_np(&((ar_struct)._array), (void **)\
                           &((ar_struct).data), (edim), (np_ptr))

static int _ar_import(ar_dsc, data, edim)
    array_descr * ar_dsc;
    void **data;
    unsigned int edim;
{
    ar_dsc->shape = NULL;
    ar_dsc->strides = NULL;
    
    PyArrayObject *_array = ar_dsc->np_array;
    unsigned int ii = 0;
    
    unsigned int _ndim = (unsigned int) PyArray_NDIM(_array);
    
    if (edim > 0 and _ndim != edim) {
        pyexc(PyExc_TypeError, "numpy array expected to be %u "
                               "dimensional but we got %u dimensional "
                               "array!", edim, _ndim);

        return 1;
    }
    
    ar_dsc->ndim = _ndim;
    
    ar_dsc->shape = PyArray_DIMS(_array);
    
    int elemsize = (int) PyArray_ITEMSIZE(_array);
    
    if ((ar_dsc->strides = PyMem_New(unsigned int, _ndim)) == NULL) {
        pyexcs(PyExc_MemoryError, "Failed to allocate memory for array "
                                  "strides!");
        return 1;
    }
    
    npy_intp * _strides = PyArray_STRIDES(_array);
    
    for(ii = 0; ii < _ndim; ++ii)
        ar_dsc->strides[ii] = (unsigned int) ((double) _strides[ii] / elemsize);

    *data = PyArray_DATA(_array);
    
    return 0;
}

#define ar_import_check(ar_struct, edim)\
        _ar_import(&((ar_struct)._array), (void **) &((ar_struct).data), (edim))

#define ar_import(ar_struct)\
        _ar_import(&((ar_struct)._array), (void **) &((ar_struct).data), 0)

void _ar_free(ar_dsc)
    array_descr * ar_dsc;
{
    PyMem_Del(ar_dsc->strides);
    ar_dsc->strides = NULL;
    
    PyMem_Del(ar_dsc->shape);
    ar_dsc->shape = NULL;
}

void _ar_xfree(ar_dsc)
    array_descr * ar_dsc;
{
    if (ar_dsc->strides != NULL) {
        PyMem_Del(ar_dsc->strides);
        ar_dsc->strides = NULL;
    }
    
    if (ar_dsc->shape != NULL) {
        PyMem_Del(ar_dsc->shape);
        ar_dsc->shape = NULL;
    }
}

#define ar_free(ar_struct) _ar_free(&((ar_struct)._array))
#define ar_xfree(ar_struct) _ar_xfree(&((ar_struct)._array))

#define ar_ndim(ar_struct) (ar_struct)._array.ndim
#define ar_stride(ar_struct, dim) (unsigned int) (ar_struct)._array.strides[(dim)]
#define ar_strides(ar_struct) (ar_struct)._array.strides
#define ar_shape(ar_struct, dim) (unsigned int) (ar_struct)._array.shape[(dim)]

#define ar_rows(ar_struct) (unsigned int) (ar_struct)._array.shape[0]
#define ar_cols(ar_struct) (unsigned int) (ar_struct)._array.shape[1]

#define ar_data(ar_struct) (ar_struct).data
#define ar_np_ptr(ar_struct) (ar_struct)._array.np_array

#define np_array(__array) &PyArray_Type, &((__array)._array.np_array)

#define ar_elem1(ar_struct, ii)\
        (ar_struct).data[(ii) * (ar_struct)._array.strides[0]]

#define ar_elem2(ar_struct, ii, jj)\
        (ar_struct).data[  (ii) * (ar_struct)._array.strides[0]\
                         + (jj) * (ar_struct)._array.strides[1]]

#define ar_elem3(ar_struct, ii, jj, kk)\
        (ar_struct).data[  (ii) * (ar_struct)._array.strides[0]\
                         + (jj) * (ar_struct)._array.strides[1]\
                         + (kk) * (ar_struct)._array.strides[2]]


#define ar_ptr1(ar_struct, ii)\
        (ar_struct).data + (ii) * (ar_struct)._array.strides[0]

#define ar_ptr2(ar_struct, ii, jj)\
        (ar_struct).data + (ii) * (ar_struct)._array.strides[0]\
                         + (jj) * (ar_struct)._array.strides[1]

#define ar_ptr3(ar_struct, ii, jj, kk)\
        (ar_struct).data + (ii) * (ar_struct)._array.strides[0]\
                         + (jj) * (ar_struct)._array.strides[1]\
                         + (kk) * (ar_struct)._array.strides[2]


/******************************
 * Function definition macros *
 ******************************/

#define py_noargs PyObject *self
#define py_varargs PyObject *self, PyObject *args
#define py_keywords PyObject *self, PyObject *args, PyObject *kwargs

#define pymeth_noargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define pymeth_varargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define pymeth_keywords(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define pydoc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

#define keywords(...) char * keywords[] = {__VA_ARGS__, NULL}

#define np_array(__array) &PyArray_Type, &((__array)._array.np_array)

#define parse_varargs(format, ...) \
do {\
    if (!PyArg_ParseTuple(args, format, __VA_ARGS__))\
        return NULL;\
} while(0)

#define parse_keywords(format, ...) \
do {\
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,\
                                     __VA_ARGS__))\
        return NULL;\
} while(0)


/********************************
 * Module initialization macros *
 ********************************/

#define init_methods(module_name, ...)\
static PyMethodDef CONCAT(module_name, _methods)[] = {\
    __VA_ARGS__,\
    {NULL, NULL, 0, NULL}\
};


// Python 3
#if PY_VERSION_HEX >= 0x03000000

#define PyString_Check PyBytes_Check
#define PyString_GET_SIZE PyBytes_GET_SIZE
#define PyString_AS_STRING PyBytes_AS_STRING
#define PyString_FromString PyBytes_FromString
#define PyUString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_ConcatAndDel PyBytes_ConcatAndDel
#define PyString_AsString PyBytes_AsString

#define PyInt_Check PyLong_Check
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AS_LONG PyLong_AsLong
#define PyInt_AsLong PyLong_AsLong

#define PyNumber_Int PyNumber_Long

#define RETVAL m

#define init_module(module_name, module_doc, version)\
static struct PyModuleDef CONCAT(module_name, _moduledef) = {\
    PyModuleDef_HEAD_INIT,\
    QUOTE(module_name),\
    NULL,\
    -1,\
    CONCAT(module_name, _methods),\
    NULL,\
    NULL,\
    NULL,\
    NULL\
};\
\
static PyObject * CONCAT(module_name, _error);\
static PyObject * CONCAT(module_name, _module);\
PyMODINIT_FUNC CONCAT(PyInit_, module_name)(void) {\
    PyObject *m,*d, *s;\
    \
    m = CONCAT(module_name, _module) = \
    PyModule_Create(&(CONCAT(module_name, _moduledef)));\
    \
    import_array();\
    \
    if (PyErr_Occurred()) {\
        PyErr_SetString(PyExc_ImportError, "can't initialize module "\
                                           QUOTE(module_name) \
                                           "(failed to import numpy)");\
        return RETVAL;\
    }\
    \
    d = PyModule_GetDict(m);\
        s = PyString_FromString("$Revision: " QUOTE(version) " $");\
    \
    PyDict_SetItemString(d, "__version__", s);\
    \
    s = PyUnicode_FromString(module_doc);\
    \
    PyDict_SetItemString(d, "__doc__", s);\
    \
    CONCAT(module_name, _error) = PyErr_NewException(QUOTE(module_name)".error",\
                                                     NULL, NULL);\
    \
    Py_DECREF(s);\
    \
    return RETVAL;\
}

#else // Python 2

#define PyUString_FromStringAndSize PyString_FromStringAndSize

#define RETVAL

#define init_module(module_name, module_doc, version)\
static PyObject * CONCAT(module_name, _error);\
static PyObject * CONCAT(module_name, _module);\
PyMODINIT_FUNC CONCAT(init, module_name)(void) {\
    PyObject *m,*d, *s;\
    \
    import_array();\
    \
    m = inmet_aux_module = Py_InitModule(QUOTE(module_name),\
                                         CONCAT(module_name, _methods));\
    \
    if (PyErr_Occurred()) {\
        PyErr_SetString(PyExc_ImportError, "can't initialize module "\
                                           QUOTE(module_name) \
                                           "(failed to import numpy)");\
        return RETVAL;\
    }\
    \
    d = PyModule_GetDict(m);\
    \
    s = PyString_FromString("$Revision: " QUOTE(version) " $");\
    \
    PyDict_SetItemString(d, "__version__", s);\
    \
    s = PyString_FromString(module_doc);\
    \
    PyDict_SetItemString(d, "__doc__", s);\
    \
    CONCAT(module_name, _error) = PyErr_NewException(QUOTE(module_name)".error",\
                                                     NULL, NULL);\
    Py_DECREF(s);\
    return RETVAL;\
}

#endif // Python 2/3

#endif
