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

#ifndef CAPI_FUNCTIONS_H
#define CAPI_FUNCTIONS_H

#include "Python.h"
#include "numpy/arrayobject.h"

typedef PyArrayObject* np_ptr;
typedef PyObject* py_ptr;

// turn s into string "s"
#define QUOTE(s) # s

/*************
 * IO macros *
 *************/

#define py_error(text) PySys_WriteStderr(text)
#define py_errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define py_print(string) PySys_WriteStdout(string)
#define py_prints(format, ...) PySys_WriteStdout(format, __VA_ARGS__)
#define py_println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define py_log println("File: %s line: %d", __FILE__, __LINE__)

#define pyexc(exc_type, format, ...)\
        PyErr_Format((exc_type), (format), __VA_ARGS__)\

#define pyexcs(exc_type, string)\
        PyErr_Format((exc_type), (string))\

/************************************
 * Wrapper object for numpy arrays. *
 ************************************/
 
typedef struct array_descr_t {
    size_t ndim;
    npy_intp *strides, *shape;
} array_descr;    

typedef struct ar_double_t {
    npy_double * data;
    array_descr _array;
} ar_double;

typedef struct ar_bool_t {
    npy_bool * data;
    array_descr _array;
} ar_bool;


static int _ar_setup_array_dsc(np_ptr array, array_descr * ar_dsc,
                               const char * name)
{
    int ii = 0;
    int array_ndim = PyArray_NDIM(array);
    int elemsize = (int) PyArray_ITEMSIZE(array);
    
    ar_dsc->ndim = (size_t) array_ndim;
    ar_dsc->shape = PyArray_DIMS(array);
    if ((ar_dsc->strides = PyMem_New(npy_intp, array_ndim)) == NULL) {
        pyexc(PyExc_MemoryError, "Could not allocate memory for storing the "
                                 "strides of %s array.", name);
        
        return 1;
    }
    
    npy_intp * tmp = PyArray_STRIDES(array);
    
    for(ii = 0; ii < array_ndim; ++ii)
        ar_dsc->strides[ii] = tmp[ii] / elemsize;
    
    return 0;
}


#define ar_setup(ar_struct, np_array)\
do {\
    if( _ar_setup_array_dsc(np_array, &((ar_struct)._array)),\
                            QUOTE((ar_struct)) )\
        goto fail;\
    \
} while(0)

#define ar_free(ar_struct) PyMem_Del((ar_struct)._array.strides)

#define ar_xfree(ar_struct)\
do {\
    if ((ar_struct)._array.strides != NULL)\
        PyMem_Del((ar_struct)._array.strides);\
} while(0)

#define ar_ndim(ar_struct) (ar_struct)._array.ndim
#define ar_stride(ar_struct, dim) (size_t) (ar_struct)._array.strides[dim]
#define ar_strides(ar_struct) (ar_struct)._array.strides
#define ar_shape(ar_struct, dim) (size_t) (ar_struct)._array.shape[dim]

#define ar_rows(ar_struct) ar_shape((ar_struct), 0)
#define ar_cols(ar_struct) ar_shape((ar_struct), 1)

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

#define np_check_type(array, tp)\
do {\
    if (PyArray_TYPE((array)) != (tp)) {\
        pyexc(PyExc_TypeError, "%s array is not of correct type (%d)",\
                               QUOTE((array)), (tp));\
        goto fail;\
    }\
} while(0)

#define np_check_callable(func)\
do {\
    if (!PyCallable_Check(func)) {\
        pyexc(PyExc_TypeError, "%s is not a callable function", QUOTE(func));\
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
