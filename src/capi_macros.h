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

#include "params_types.h"


int _convert_array_check(np_ptr array, const py_ptr to_convert, const int typenum,
                         const int requirements, const int ndim,
                         const char * name);

int _check_matrix(const np_ptr array, const int rows, const int cols,
                  const char * name);

int _check_ndim(const np_ptr array, const int ndim, const char * name);


int _check_dim(const np_ptr array, const int dim, const int expected_length,
               const char * name);

// turn s into string "s"
#define QUOTE(s) # s

/****************************
 * Python 2/3 Compatibility *
 ****************************/


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
#define init_module(module_name, module_doc, ...)\
    static PyMethodDef module_name ## _methods[] = {\
        __VA_ARGS__,\
        {"_error_out", (PyCFunction)error_out, METH_NOARGS, NULL},\
        {NULL, NULL, 0, NULL}\
    };\
    \
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

/************
 * Python 2 *
 ************/
#else

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
#define init_module(module_name, ...)\
    static  PyMethodDef module_name ## _methods[] = {\
    __VA_ARGS__,\
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},\
    {NULL, NULL, 0, NULL}\
    };\

    void init ## module_name(void)\
    {\
        import_array();
        PyObject *module = Py_InitModule(QUOTE(module_name), module_name ## _methods);\
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


/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/*************
 * IO macros *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define Log print("File: %s line: %d", __FILE__, __LINE__)

/******************************
 * Function definition macros *
 ******************************/

#define py_varargs PyObject * self, PyObject * args
#define py_keywords PyObject * self, PyObject * args, PyObject * kwargs

//----------------------------------------------------------------------

#define pymeth_noargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define pymeth_varargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define pymeth_keywords(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define pyfun_doc(fun_name, doc) PyDoc_VAR( fun_name ## __doc__) = PyDoc_STR(doc)

#define pyfun_parse_keywords(keywords, format, ...) \
do {\
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,\
                                     __VA_ARGS__))\
        return NULL;\
} while(0)

#define pyfun_parse_varargs(format, ...) \
do {\
    if (!PyArg_ParseTuple(args, format, __VA_ARGS__))\
        return NULL;\
} while(0)

/****************************
 * Numpy convenience macros *
 ****************************/

#define np_ptr1(obj, ii) PyArray_GETPTR1((obj), (ii))
#define np_gptr(obj, ii, jj) PyArray_GETPTR2((obj), (ii), (jj))
#define np_dim(obj, idx) PyArray_DIM((obj), (idx))
#define np_ndim(obj, idx) PyArray_NDIM((obj), (idx))
#define np_delem(obj, ii, jj) *((npy_double *) PyArray_GETPTR2((obj), (ii), (jj)))
#define np_belem1(obj, ii) *((npy_bool *) PyArray_GETPTR1((obj), (ii)))

#define np_data(obj) PyArray_DATA((obj))

/* based on
 * https://github.com/sniemi/SamPy/blob/master/sandbox/src1/TCSE3-3rd-examples
 * /src/C/NumPy_macros.h */

#define np_import(array_out, array_to_convert, typenum, requirements, name)\
do {\
    if ((array_out) = (np_ptr) PyArray_FROM_OTF(to_convert, typenum,\
                                                requirements) == NULL) {\
        PyErr_Format(PyExc_ValueError, "Failed to import array %s",\
                     (name));\
        goto fail;\
    }\
} while(0)

#define np_import_check(array_out, array_to_convert, typenum, requirements,\
                        ndim, name)\
do {\
    if (_convert_array_check((array_out), (array_to_convert), (typenum),\
                           (requirements), (ndim), (name))) {\
        goto fail;\
    }\
} while(0)

#define np_check_matrix(array, rows, cols, name)\
do {\
    if (_check_matrix((array), (rows), (cols), (name)))\
        goto fail;\
} while(0)

#define np_empty(array_out, ndim, shape, typenum, is_fortran)\
do {\
    (array_out) = (np_ptr) PyArray_EMPTY((ndim), (shape), (typenum),\
                                          (is_fortran));\
    if ((array_out) == NULL) {\
        PyErr_Format(PyExc_ValueError, "Failed to create empty array %s",\
                     QUOTE((array_out)));\
        goto fail;\
    }\
} while(0)

#define np_check_ndim(array, ndim, name)\
do {\
    if (_check_ndim((array), (ndim), (name)))\
        goto fail;\
} while(0)

#define np_check_dim(array, dim, expected_length, name)\
do {\
    if (_check_dim((array), (dim), (expected_length), (name)))\
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

/***************
 * ERROR CODES *
 ***************/

#define IO_ERR    -1
#define ALLOC_ERR -2
#define NUM_ERR   -3

#endif
