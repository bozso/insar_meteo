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
    const int typenum;
    uint ndim;
    npy_intp *strides, *shape;
    np_ptr np_array;
    const char * name;
} array_descr;    

typedef struct ar_double_t {
    npy_double * data;
    array_descr _array;
} _ar_double;

typedef struct ar_bool_t {
    npy_bool * data;
    array_descr _array;
} _ar_bool;

#define init_array(type, nname) {._array = {.typenum = (type), .name = (nname) } }

#define ar_double(x) _ar_double (x) = init_array(NPY_DOUBLE, QUOTE((x)))
#define ar_bool(x) _ar_bool x = init_array(NPY_BOOL, QUOTE((x)))

static int _setup_ar_dsc(np_ptr array, array_descr * ar_dsc)
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
    
    return 0;
}


static int _ar_import_check(array_descr * ar_dsc, void ** ptr,
                            const py_ptr to_convert, const int requirements,
                            const int ndim)
{
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_FROM_OTF(to_convert, ar_dsc->typenum, requirements))
         == NULL)
         return 1;
    
    int array_ndim = PyArray_NDIM(tmp);
    
    if (ndim > 0) {
        if (array_ndim != ndim) {
            PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                           "expected to be %d-dimensional",
                                           ar_dsc->name, array_ndim, ndim);
            return 1;
        }
    }
    
    _setup_ar_dsc(tmp, ar_dsc);
    
    *ptr = PyArray_DATA(tmp);
    
    return 0;
}

#define ar_import_check(ar_struct, to_convert, edim)\
do {\
    if(_ar_import_check(&((ar_struct)._array), (void **) &((ar_struct).data),\
                        (to_convert), NPY_ARRAY_IN_ARRAY, (edim)))\
        goto fail;\
    \
} while(0)

#define ar_import(ar_struct, to_convert)\
        ar_import_check((ar_struct), (to_convert), 0)

static int _ar_empty_cf(array_descr * ar_dsc, void ** ptr, const int edim,
                        npy_intp * shape, const int is_fortran)
{
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_EMPTY(edim, shape, ar_dsc->typenum, is_fortran))
         == NULL) {
        PyErr_Format(PyExc_ValueError, "Failed to create empty array: %s",
                                        ar_dsc->name);
        return 1;
    }
    
    _setup_ar_dsc(tmp, ar_dsc);
    *ptr = PyArray_DATA(tmp);
    
    return 0;
}

#define ar_empty_cf(ar_struct, edim, shape, is_fortran)\
do {\
    if(_ar_empty_cf(&((ar_struct)._array), (void **) &((ar_struct).data),\
                        (edim), (shape), (is_fortran)))\
        goto fail;\
    \
} while(0)

#define ar_empty(ar_struct, edim, shape)\
        ar_empty_cf((ar_struct), (edim), (shape), 0)

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

#define ar_return(ar_struct) PyArray_Return((ar_struct)._array.np_array)

#define ar_ndim(ar_struct) (ar_struct)._array.ndim
#define ar_stride(ar_struct, dim) (uint) (ar_struct)._array.strides[dim]
#define ar_strides(ar_struct) (ar_struct)._array.strides
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

#define py_noargs py_ptr self
#define py_varargs py_ptr self, py_ptr args
#define py_keywords py_ptr self, py_ptr args, py_ptr kwargs

#define pymeth_noargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define pymeth_varargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define pymeth_keywords(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define pydoc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

#define keywords(...) char * keywords[] = {__VA_ARGS__, NULL}

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

#endif
