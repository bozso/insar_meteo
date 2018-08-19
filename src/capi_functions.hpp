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

#include <cstring>
#include <type_traits>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "utils.hpp"

typedef PyArrayObject * np_ptr;
typedef PyObject * py_ptr;

#define pyexcf(exc_type, format, ...)\
        PyErr_Format((exc_type), (format), __VA_ARGS__)

#define pyexc(exc_type, format)\
        PyErr_Format((exc_type), (format))

/*************
 * IO macros *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
#define prints(string, ...) PySys_WriteStdout(string, __VA_ARGS__)
#define println(format, ...) PySys_WriteStdout(format"\n", __VA_ARGS__)

#define _log println("File: %s line: %d", __FILE__, __LINE__)

template<class T>
class np_wrap
{
    private:
        uint ndim;
        npy_intp *strides, *shape;
        np_ptr np_array;
        T * data;
        char * name;
        bool decrefd;
    public:
        void convert_np_array(const np_ptr tmp, const char * nname);
        
        void import(const py_ptr to_convert, const char * nname,
                    const int edim = 0);
        
        void empty(int ndim, npy_intp * shape, int is_fortran = 0,
                   const char * name = "");
        
        void decref(void);
        void xdecref(void);
        
        void check_matrix(uint rows, uint cols);
        void check_rows(uint rows);
        void check_cols(uint cols);
        
        T * get_data();
        np_ptr get_array();

        T& operator()(uint ii);
        T& operator()(uint ii, uint jj);
        T& operator()(uint ii, uint jj, uint kk);

        const uint rows();
        const uint cols();
        const uint get_shape(uint ii);
        const uint get_stride(uint ii);

        ~np_wrap();
};

template<class T>
inline const int np_type()
{
    if (std::is_same<T, npy_double>::value)
        return NPY_DOUBLE;

    else if (std::is_same<T, npy_bool>::value)
        return NPY_BOOL;

    else if (std::is_same<T, npy_byte>::value)
        return NPY_BYTE;
    else if (std::is_same<T, npy_ubyte>::value)
        return NPY_UBYTE;

    else if (std::is_same<T, npy_short>::value)
        return NPY_SHORT;
    else if (std::is_same<T, npy_ushort>::value)
        return NPY_USHORT;

    else if (std::is_same<T, npy_int>::value)
        return NPY_INT;
    else if (std::is_same<T, npy_uint>::value)
        return NPY_UINT;

    else if (std::is_same<T, npy_long>::value)
        return NPY_LONG;
    else if (std::is_same<T, npy_ulong>::value)
        return NPY_ULONG;

    else if (std::is_same<T, npy_longlong>::value)
        return NPY_LONGLONG;
    else if (std::is_same<T, npy_ulonglong>::value)
        return NPY_ULONGLONG;

    else if (std::is_same<T, npy_float16>::value)
        return NPY_FLOAT16;

    else if (std::is_same<T, npy_float32>::value)
        return NPY_FLOAT32;

    else if (std::is_same<T, npy_float64>::value)
        return NPY_FLOAT64;

    else if (std::is_same<T, npy_longdouble>::value)
        return NPY_LONGDOUBLE;

    else if (std::is_same<T, npy_complex64>::value)
        return NPY_COMPLEX64;

    else if (std::is_same<T, npy_complex128>::value)
        return NPY_COMPLEX128;

    else if (std::is_same<T, npy_clongdouble>::value)
        return NPY_CLONGDOUBLE;

    else {
        pyexc(PyExc_ValueError, "Unknown numpy type passed to np_type!");
        throw "Unrecognized numpy type!";
    }
}


template<class T>
inline void np_wrap<T>::convert_np_array(const np_ptr tmp, const char * nname)
{
    uint array_ndim = (uint) PyArray_NDIM(tmp);
    
    double elemsize = (double) PyArray_ITEMSIZE(tmp);
    
    ndim = array_ndim;
    shape = PyArray_DIMS(tmp);
    strides = PyMem_New(npy_intp, array_ndim);
    
    npy_intp * _strides = PyArray_STRIDES(tmp);
    
    for(uint ii = 0; ii < array_ndim; ++ii)\
        strides[ii] = (npy_intp) (_strides[ii] / elemsize);
    
    data = (T*) PyArray_BYTES(tmp);
    np_array = tmp;
    
    std::memcpy(name, nname, strlen(nname + 1));
    
    decrefd = false;
}


template<class T>
inline void np_wrap<T>::import(const py_ptr to_convert, const char * nname,
                               const int edim)
{
    static const int typenum = np_type<T>();
    
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, NPY_ARRAY_IN_ARRAY))
         == nullptr) {
        _log;
        decrefd = true;
        _log;
        throw "Failed to convert to numpy array!";
        _log;
    }
    
    int array_ndim = PyArray_NDIM(tmp);
    
    if (edim > 0) {
        if (array_ndim != edim) {
            pyexcf(PyExc_ValueError, "Array %s is %d-dimensional, but expected "
                                     "to be %d-dimensional", nname, array_ndim,
                                     edim);
            throw "Wrong number of dimensions!";
        }
    }
    convert_np_array(tmp, nname);
}


template<class T>
inline void np_wrap<T>::empty(int ndim, npy_intp * shape, int is_fortran,
                              const char * name)
{
    static const int typenum = np_type<T>();

    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_EMPTY(ndim, shape, typenum, is_fortran))
         == nullptr) {
        pyexcf(PyExc_RuntimeError, "Could not create empty array %s!", name);
        decrefd = true;
        throw "Array creation failed!";
        
    }
    
    convert_np_array(tmp, name);
}


template<class T>
inline void np_wrap<T>::decref()
{
    decrefd = true;
    Py_DECREF(np_array);
    PyMem_Del(strides);
}


template<class T>
inline void np_wrap<T>::xdecref()
{
    decrefd = true;
    Py_XDECREF(np_array);
    
    if (strides != nullptr)
        PyMem_Del(strides);
}

template<class T>
np_wrap<T>::~np_wrap()
{
    if (!decrefd) {
        Py_XDECREF(np_array);
        
        if (strides != nullptr)
            PyMem_Del(strides);
    }
}        

template<class T>
inline void np_wrap<T>::check_matrix(uint rows, uint cols)
{
    uint tmp = (uint) shape[0];
    if (rows != tmp) {
        pyexcf(PyExc_ValueError, "Array %s has wrong number of rows=%d "\
                                       "(expected %d)", name, rows, tmp);
        throw "Wrong number of rows!";
    }

    tmp = (uint) shape[1];
    if (cols != tmp) {
        pyexcf(PyExc_ValueError, "Array %s has wrong number of cols=%d "\
                                       "(expected %d)", name, cols, tmp);
        throw "Wrong number of cols!";
    }
}

template<class T>
inline void np_wrap<T>::check_rows(uint rows)
{
    uint tmp = (uint) shape[0];
    if (rows != tmp) {
        pyexcf(PyExc_ValueError, "Array %s has wrong number of rows=%d "\
                                       "(expected %d)", name, rows, tmp);
        throw "Wrong number of rows!";
    }
}

template<class T>
inline void np_wrap<T>::check_cols(uint cols)
{
    uint tmp = (uint) shape[0];
    if (rows != tmp) {
        pyexcf(PyExc_ValueError, "Array %s has wrong number of cols=%d "\
                                       "(expected %d)", name, cols, tmp);
        throw "Wrong number of cols!";
    }
}

template<class T>
inline T* np_wrap<T>::get_data()
{
    return data;
}

template<class T>
inline np_ptr np_wrap<T>::get_array()
{
    return np_array;
}

template<class T>
inline const uint np_wrap<T>::rows()
{
    return static_cast<uint>(shape[0]);
}


template<class T>
inline const uint np_wrap<T>::cols()
{
    return static_cast<uint>(shape[1]);
}


template<class T>
inline const uint np_wrap<T>::get_shape(uint ii)
{
    return static_cast<uint>(shape[ii]);
}


template<class T>
inline const uint np_wrap<T>::get_stride(uint ii)
{
    return static_cast<uint>(strides[ii]);
}


template<class T>
inline T& np_wrap<T>::operator()(uint ii)
{
    return data[ii * strides[0]];
}


template<class T>
inline T& np_wrap<T>::operator()(uint ii, uint jj)
{
    return data[ii * strides[0] + jj * strides[1]];
}


template<class T>
inline T& np_wrap<T>::operator()(uint ii, uint jj, uint kk)
{
    return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
}

// turn s into string "s"
#define QUOTE(s) # s

/***********************************************************
 * Python 2/3 compatible module initialization biolerplate *
 ***********************************************************/
#define init_table(module_name, ...)\
    static PyMethodDef module_name ## _methods[] = {\
        __VA_ARGS__,\
        {"_error_out", (PyCFunction)error_out, METH_NOARGS, NULL},\
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
#define init_module(module_name, module_doc, ...)\
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
#define init_module(module_name, module_doc, ...)\
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

#define pyfun_doc(fun_name, doc) PyDoc_VAR(fun_name ## __doc__) = PyDoc_STR(doc)

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

#endif
