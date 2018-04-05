#ifndef CAPI_MACROS_H
#define CAPI_MACROS_H

//----------------------------------------------------------------------
// WGS-84 ELLIPSOID PARAMETERS
//----------------------------------------------------------------------

// RADIUS OF EARTH
#define R_earth 6372000

#define WA  6378137.0
#define WB  6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2  6.694380e-03


//----------------------------------------------------------------------
// DEGREES, RADIANS, PI
//----------------------------------------------------------------------

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01


#define distance(x, y, z) sqrt((y)*(y)+(x)*(x)+(z)*(z))

#define OR ||
#define AND &&

//----------------------------------------------------------------------
// FOR MACROS
// REQUIRES C99 standard!
//----------------------------------------------------------------------

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

//----------------------------------------------------------------------
// IO MACROS
//----------------------------------------------------------------------

#define error(text) PySys_WriteStderr(text"\n")
#define errora(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)
#define print(format, ...) PySys_WriteStdout(format, __VA_ARGS__)

#define _log print("%s\t%d\n", __FILE__, __LINE__)

//----------------------------------------------------------------------
// FUNCTION DEFINITION MACROS
//----------------------------------------------------------------------

#define PyFun_Noargs(fun_name) static PyObject * fun_name (PyObject * self)

#define PyFun_Varargs(fun_name) \
static PyObject * fun_name (PyObject * self, PyObject * args)

#define PyFun_Keywords(fun_name) \
static PyObject * fun_name (PyObject * self, PyObject * args, PyObject * kwargs)

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
({ \
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, \
                                     __VA_ARGS__))\
        return NULL; \
})

//----------------------------------------------------------------------
// NUMPY CONVENIENCE MACROS
//----------------------------------------------------------------------

#define NPY_AO PyArrayObject 
#define NPY_Ptr1(obj, i) PyArray_GETPTR1(obj, ii)
#define NPY_Ptr(obj, ii, jj) PyArray_GETPTR2(obj, ii, jj)
#define NPY_Dim(obj, idx) PyArray_DIM(obj, idx)
#define NPY_Ndim(obj, idx) PyArray_NDIM(obj, idx)
#define NPY_Delem(obj, ii, jj) *((npy_double *) PyArray_GETPTR2(obj, ii, jj))

//----------------------------------------------------------------------
// ERROR CODES
//----------------------------------------------------------------------

#define IO_ERR    -1
#define ALLOC_ERR -2
#define NUM_ERR   -3

#endif
