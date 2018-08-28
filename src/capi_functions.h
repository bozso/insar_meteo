/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CAPI_MACROS_H
#define CAPI_MACROS_H

#include "Python.h"
#include "numpy/arrayobject.h"

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;
typedef const double cdouble;
typedef unsigned int uint;

// turn s into string "s"
#define QUOTE(s) # s

/*************
 * IO macros *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define _log println("File: %s line: %d", __FILE__, __LINE__)

/************************************
 * Wrapper object for numpy arrays. *
 ************************************/
 
typedef struct array_descr_t {
    uint ndim;
    npy_intp *strides, *shape;
    np_ptr np_array;
    char * name;
} array_descr;    

typedef struct ar_double_t {
    npy_double * data;
    array_descr _array;
} ar_double;

typedef struct ar_bool_t {
    npy_bool * data;
    array_descr _array;
} ar_bool;

static int _setup_ar_dsc(np_ptr array, array_descr * ar_dsc, char * name)
{
    int array_ndim = PyArray_NDIM(array);
    int elemsize = (int) PyArray_ITEMSIZE(array);
    
    ar_dsc->ndim = (uint) array_ndim;
    ar_dsc->shape = PyArray_DIMS(array);
    ar_dsc->strides = PyMem_New(npy_intp, array_ndim);
    
    npy_intp * tmp = PyArray_STRIDES(array);
    
    for(int ii = 0; ii < array_ndim; ++ii)
        ar_dsc->strides[ii] = tmp[ii] / elemsize;
    
    ar_dsc->np_array = array;
    ar_dsc->name = name;
    
    return 0;
}


static int _ar_import_check(array_descr * ar_dsc, void ** ptr,
                            const py_ptr to_convert, const int typenum,
                            const int requirements, const int ndim,
                            char * name)
{
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, requirements))
         == NULL)
         return 1;
    
    int array_ndim = PyArray_NDIM(tmp);
    
    if (ndim > 0) {
        if (array_ndim != ndim) {
            PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                           "expected to be %d-dimensional", name,
                                            array_ndim, ndim);
            return 1;
        }
    }
    
    _setup_ar_dsc(tmp, ar_dsc, name);
    
    *ptr = PyArray_DATA(tmp);
    
    return 0;
}

#define ar_import_check(ar_struct, to_convert, typenum, edim, nname)\
do {\
    if(_ar_import_check(&((ar_struct)._array), (void **) &((ar_struct).data),\
                        (to_convert), (typenum), NPY_ARRAY_IN_ARRAY, (edim),\
                        (nname)))\
        goto fail;\
    \
} while(0)

#define ar_import(ar_struct, to_convert, typenum, name)\
        ar_import_check((ar_struct), (to_convert), (typenum), 0, (name))

#define ar_import_check_double(ar_struct, to_convert, ndim, name)\
        ar_import_check((ar_struct), (to_convert), NPY_DOUBLE, (ndim), (name))

static int _ar_empty_cf(array_descr * ar_dsc, void ** ptr, const int edim,
                        npy_intp * shape, const int typenum,
                        const int is_fortran, char * name)
{
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_EMPTY(edim, shape, typenum, is_fortran))
         == NULL) {
        PyErr_Format(PyExc_ValueError, "Failed to create empty array: %s",
                                        name);
        return 1;
    }
    
    _setup_ar_dsc(tmp, ar_dsc, name);
    *ptr = PyArray_DATA(tmp);
    
    return 0;
}

#define ar_empty_cf(ar_struct, edim, shape, typenum, is_fortran, name)\
do {\
    if(_ar_empty_cf(&((ar_struct)._array), (void **) &((ar_struct).data),\
                        (edim), (shape), (typenum), (is_fortran), (name)))\
        goto fail;\
    \
} while(0)

#define ar_empty(ar_struct, edim, shape, typenum, name)\
        ar_empty_cf((ar_struct), (edim), (shape), (typenum), 0, (name))

#define ar_empty_double(ar_struct, edim, shape, name)\
        ar_empty_cf((ar_struct), (edim), (shape), NPY_DOUBLE, 0, (name))

#define ar_empty_bool(ar_struct, edim, shape, name)\
        ar_empty_cf((ar_struct), (edim), (shape), NPY_BOOL, 0, (name))

#define ar_decref(ar_struct)\
do {\
    Py_DECREF((ar_struct)._array.np_array);\
    PyMem_Del((ar_struct)._array.strides);\
} while(0)

#define ar_xdecref(ar_struct)\
do {\
    Py_XDECREF((ar_struct)._array.np_array);\
    if ((ar_struct)._array.strides != NULL)\
        PyMem_Del((ar_struct)._array.strides);\
} while(0)

#define ar_np_array(ar_struct) (ar_struct)._array.np_array

#define ar_ndim(ar_struct) (ar_struct)._array.ndim
#define ar_stride(ar_struct, dim) (uint) (ar_struct)._array.strides[dim]
#define ar_strides(ar_struct) (uint) (ar_struct)._array.strides
#define ar_dim(ar_struct, dim) (uint) (ar_struct)._array.shape[dim]

#define ar_rows(ar_struct) ar_dim((ar_struct), 0)
#define ar_cols(ar_struct) ar_dim((ar_struct), 1)

#define ar_data(ar_struct) (ar_struct).data

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

static int _ar_check_matrix(array_descr * ar_dsc, const uint rows,
                            const uint cols)
{
    uint tmp = (uint) ar_dsc->shape[0];
    if (rows != tmp) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of rows=%d "
                                       "(expected %d)", ar_dsc->name, tmp, rows);
        return 1;
    }
    
    tmp = (uint) ar_dsc->shape[1];
    if (cols != tmp) {\
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of cols=%d "
                                       "(expected %d)", ar_dsc->name, tmp, cols);
        return 1;
    }
    
    return 0;
}

#define ar_check_matrix(ar_struct, rows, cols)\
do {\
    if(_ar_check_matrix(&((ar_struct)._array), rows, cols))\
        goto fail;\
} while(0)

static int _ar_check_rows(array_descr * ar_dsc, const uint rows)\
{
    uint tmp = (uint) ar_dsc->shape[0];
    if (rows != tmp) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of rows=%d "
                                       "(expected %d)", ar_dsc->name, tmp, rows);
        return 1;
    }
    return 0;
}

static int _ar_check_cols(array_descr * ar_dsc, const uint cols)\
{
    uint tmp = (uint) ar_dsc->shape[1];
    if (cols != tmp) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of cols=%d "
                                       "(expected %d)", ar_dsc->name, tmp, cols);
        return 1;
    }
    return 0;
}

#define ar_check_rows(ar_struct, rows)\
do {\
    if(_ar_check_rows(&((ar_struct)._array), rows))\
        goto fail;\
} while(0)

#define ar_check_cols(ar_struct, cols)\
do {\
    if(_ar_check_cols(&((ar_struct)._array), cols))\
        goto fail;\
} while(0)


#define np_check_type(array, tp)\
do {\
    if (PyArray_TYPE(array) != (tp)) {\
        PyErr_Format(PyExc_TypeError,\
        "%s array is not of correct type (%d)", QUOTE(array), (tp));\
        goto fail;\
    }\
} while(0)

#define np_check_callable(func)\
do {\
    if (!PyCallable_Check(func)) {\
        PyErr_Format(PyExc_TypeError,\
        "%s is not a callable function", QUOTE(func));\
        goto fail;\
    }\
} while(0)


/******************************
 * Function definition macros *
 ******************************/

#define pymeth_noargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define pymeth_varargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define pymeth_keywords(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define pyfun_doc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

#define parse_varargs(args, format, ...) \
do {\
    if (!PyArg_ParseTuple((args), format, __VA_ARGS__))\
        return NULL;\
} while(0)

#define parse_keywords(args, kwargs, keywords, format, ...) \
do {\
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,\
                                     __VA_ARGS__))\
        return NULL;\
} while(0)


/***********************************************************
 * Python 2/3 compatible module initialization biolerplate *
 ***********************************************************/

#define init_table(module_name, ...)\
static PyMethodDef module_name ## _methods[] = {\
    __VA_ARGS__,\
    {"_error_out", (PyCFunction) error_out, METH_NOARGS, NULL},\
    {NULL, NULL, 0, NULL}\
}

/************
 * Python 3 *
 ************/
#if PY_MAJOR_VERSION >= 3

#define IS_PY3K

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

struct module_state {
    PyObject *error;
};

static PyObject * error_out(PyObject *m)
{
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static int extension_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int extension_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

// for some reason only works without: do { ... } while(0)
#define init_module(module_name, module_doc)\
    static struct PyModuleDef module_name ## _moduledef = {\
        PyModuleDef_HEAD_INIT,\
        QUOTE(module_name),\
        (module_doc),\
        sizeof(struct module_state),\
        module_name ## _methods,\
        NULL,\
        extension_traverse,\
        extension_clear,\
        NULL\
    };\
    \
    PyMODINIT_FUNC PyInit_ ## module_name(void)\
    {\
        import_array();\
        PyObject *module = PyModule_Create(&(module_name ## _moduledef));\
        \
        if (module == NULL)\
            return NULL;\
        struct module_state *st = GETSTATE(module);\
        \
        st->error = PyErr_NewException(QUOTE(module_name)".Error", NULL, NULL);\
        if (st->error == NULL) {\
            Py_DECREF(module);\
            return NULL;\
        }\
        \
        return module;\
    }\

#else

/************
 * Python 2 *
 ************/

#define GETSTATE(m) (&_state)

struct module_state {
    PyObject *error;
};

static struct module_state _state;

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

// for some reason only works without: do { ... } while(0)
#define init_module(module_name, module_doc)\
    void init ## module_name(void)\
    {\
        import_array();
        PyObject *module = Py_InitModule3(QUOTE(module_name),\
                                          module_name ## _methods, (module_doc));\
        if (module == NULL)\
            return;\
        struct module_state *st = GETSTATE(module);\
        \
        st->error = PyErr_NewException(QUOTE(module_name)".Error", NULL, NULL);\
        if (st->error == NULL) {\
            Py_DECREF(module);\
            return;\
        }\
    }\

#endif

#endif
