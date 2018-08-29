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

#ifndef NUMPY_CAPI_AUX_H
#define NUMPY_CAPI_AUX_H

// turn s into string "s"
#define QUOTE(s) # s

#include "Python.h"
#include "numpy/arrayobject.h"

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

/*************
 * IO macros *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define _log println("File: %s line: %d", __FILE__, __LINE__)

static int _np_convert_array_check(np_ptr * array, const py_ptr to_convert,
                                   const int typenum, const int requirements,
                                   const int ndim, const char * name)
{
    if ((*array = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, requirements))
         == NULL)
         return 1;

    int array_ndim = PyArray_NDIM(*array);
    
    if (ndim > 0 && array_ndim != ndim) {
        PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                       "expected to be %d-dimensional", name,
                                        array_ndim, ndim);
        return 1;
    }
    
    return 0;
}

// Import a numpy array and check the number of its dimensions.
#define np_import_check(array_out, array_to_convert, typenum, requirements,\
                        ndim)\
do {\
    if (_np_convert_array_check(&(array_out), (array_to_convert), (typenum),\
                           (requirements), (ndim), QUOTE((array_out)))) {\
        goto fail;\
    }\
} while(0)

#define np_import(array_out, array_to_convert, typenum, requirements)\
        np_import_check((array_out), (array_to_convert), (typenum), (requirements), 0)\

static int _np_check_matrix(const np_ptr array, const int rows, const int cols,
                            const char * name)
{
    int tmp = PyArray_DIM(array, 0);

    if (tmp != rows) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of rows=%d "
                                       "(expected %d)", name, tmp, rows);
        return 1;
    }                                                                       

    tmp = PyArray_DIM(array, 1);

    if (tmp != cols) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of cols=%d "
                                       "(expected %d)", name, tmp, cols);
        return 1;
    }                                                                       
    
    return 0;
}    

// Check whether a matrix has the adequate number of rows and cols.
#define np_check_matrix(array, rows, cols)\
do {\
    if (_np_check_matrix((array), (rows), (cols), QUOTE((array))))\
        goto fail;\
} while(0)

static int _np_check_ndim(const np_ptr array, const int ndim, const char * name)
{
    int tmp = PyArray_NDIM(array);
    if (tmp != ndim) {
        PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                       "expected to be %d-dimensional!",
                                        name, tmp, ndim);
        return 1;
    }
    return 0;
}

// Check if a numpy array has the correct number of dimensions.
#define np_check_ndim(array, ndim)\
do {\
    if (_np_check_ndim((array), (ndim), QUOTE((array))))\
        goto fail;\
} while(0)

static int _np_check_dim(const np_ptr array, const int dim,
                      const int expected_length, const char * name)
{
    int tmp = PyArray_NDIM(array);
    if (dim > tmp) {
        PyErr_Format(PyExc_ValueError, "Array %s has no %d dimension "
                                       "(max dim. is %d)", name, dim, tmp);
        return 1;
    }
    
    tmp = PyArray_DIM(array, dim);
    
    if (tmp != expected_length) {
        PyErr_Format(PyExc_ValueError, "Array %s has wrong %d-dimension=%d "
                                       "(expected %d)", name, dim, tmp,
                                       expected_length);
        return 1;
    }
    return 0;
}

/* Check whether the number of elements for a given dimension
 * in a numpy array is adequate. */
#define np_check_dim(array, dim, expected_length)\
do {\
    if (_np_check_dim((array), (dim), (expected_length), QUOTE((array))))\
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

// Create an empty numpy array
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

#define np_empty_double(array_out, ndim, shape)\
        np_empty((array_out), (ndim), (shape), NPY_DOUBLE, 0)

/****************************
 * Numpy convenience macros *
 ****************************/

#define np_ddata(array) (double *) PyArray_DATA((array))

#define np_dim(array, idx) (uint) PyArray_DIM((array), (idx))
#define np_ndim(array) (uint) PyArray_NDIM((array))

#define np_rows(array) np_dim((array), 0)
#define np_cols(array) np_dim((array), 1)

#define np_data(array) PyArray_DATA((array))

// Check the number of rows or cols in a matrix.

#define np_check_rows(array, expected)\
        np_check_dim((array), 0, (expected))

#define np_check_cols(array, expected)\
        np_check_dim((array), 1, (expected))

#define np_return(array) PyArray_Return((array))

/******************************
 * Function definition macros *
 ******************************/

#define py_varargs PyObject * self, PyObject * args
#define py_keywords PyObject * self, PyObject * args, PyObject * kwargs

#define pymeth_noargs(fun_name) \
{QUOTE(fun_name), (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define pymeth_varargs(fun_name) \
{QUOTE(fun_name), (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define pymeth_keywords(fun_name) \
{QUOTE(fun_name), (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define pydoc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

#define parse_keywords(keywords, format, ...) \
do {\
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,\
                                     __VA_ARGS__))\
        return NULL;\
} while(0)

#define parse_varargs(format, ...) \
do {\
    if (!PyArg_ParseTuple(args, format, __VA_ARGS__))\
        return NULL;\
} while(0)

#endif
