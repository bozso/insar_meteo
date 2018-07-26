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

// turn s into string "s"
#define QUOTE(s) # s   

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA  6378137.0
#define WB  6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2  6.694380e-03


/************************
 * DEGREES, RADIANS, PI *
 ************************/

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01

#define distance(x, y, z) sqrt((y)*(y)+(x)*(x)+(z)*(z))

#define OR ||
#define AND &&

/**************************
 * FOR MACROS             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/*************
 * IO MACROS *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define Log print("File: %s line: %d", __FILE__, __LINE__)

/******************************
 * FUNCTION DEFINITION MACROS *
 ******************************/

#define PyFun_Varargs PyObject * self, PyObject * args
#define PyFun_Keywords PyObject * self, PyObject * args, PyObject * kwargs

//----------------------------------------------------------------------

#define PyFun_Method_Noargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_NOARGS, fun_name ## __doc__}

#define PyFun_Method_Varargs(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS, fun_name ## __doc__}

#define PyFun_Method_Keywords(fun_name) \
{#fun_name, (PyCFunction) fun_name, METH_VARARGS | METH_KEYWORDS, \
 fun_name ## __doc__}

#define PyFun_Doc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

#define PyFun_Parse_Keywords(keywords, format, ...) \
({\
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords,\
                                     __VA_ARGS__))\
        return NULL;\
})

#define PyFun_Parse_Varargs(format, ...) \
({\
    if (!PyArg_ParseTuple(args, format, __VA_ARGS__))\
        return NULL;\
})

/****************************
 * NUMPY CONVENIENCE MACROS *
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

#define np_import(array_out, array_to_convert, typenum, requirements)\
({\
    (array_out) = (np_ptr) PyArray_FROM_OTF((array_to_convert), (typenum),\
                                              (requirements));\
    if ((array_out) == NULL) goto fail;\
})

#define np_empty(array_out, ndim, shape, typenum, is_fortran)\
({\
    (array_out) = (np_ptr) PyArray_EMPTY((ndim), (shape), (typenum), \
                                          (is_fortran));\
    if ((array_out) == NULL) goto fail;\
})

#define np_check_ndim(a, expected_ndim)\
({\
  if (PyArray_NDIM((a)) != (expected_ndim)) {\
    PyErr_Format(PyExc_ValueError,\
    "%s array is %d-dimensional, but expected to be %d-dimensional",\
		 QUOTE(a), PyArray_NDIM(a), (expected_ndim));\
    goto fail;\
  }\
})

#define np_check_dim(a, dim, expected_length)\
({\
  if ((dim) > PyArray_NDIM((a))) {\
    PyErr_Format(PyExc_ValueError,\
    "%s array has no %d dimension (max dim. is %d)", QUOTE(a), (dim),\
    PyArray_NDIM(a));\
    goto fail;\
  } \
  if (PyArray_DIM((a), (dim)) != (expected_length)) {\
    PyErr_Format(PyExc_ValueError,\
    "%s array has wrong %d-dimension=%d (expected %d)", QUOTE(a), (dim),\
    PyArray_DIM(a, (dim)), (expected_length));\
    goto fail;\
  }\
})

#define np_check_type(a, tp)\
({\
  if (PyArray_TYPE(a) != (tp)) {\
    PyErr_Format(PyExc_TypeError,\
    "%s array is not of correct type (%d)", QUOTE(a), (tp));\
    goto fail;\
  }\
})

#define np_check_callable(func)\
({\
  if (!PyCallable_Check(func)) {\
    PyErr_Format(PyExc_TypeError,\
    "%s is not a callable function", QUOTE(func));\
    goto fail;\
  }\
})

/***************
 * ERROR CODES *
 ***************/

#define IO_ERR    -1
#define ALLOC_ERR -2
#define NUM_ERR   -3

#endif
