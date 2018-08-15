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

#include "params_types.hpp"
#include <cstring>
#include <stdexcept>

#define FOR(ii, start, stop) for(uint (ii) = 0; (ii) < (stop); ++(ii))

/*************
 * IO macros *
 *************/

#define error(text) PySys_WriteStderr(text)
#define errorln(text, ...) PySys_WriteStderr(text"\n", __VA_ARGS__)

#define print(string) PySys_WriteStdout(string)
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
    public:
        void import(const py_ptr to_convert, const int typenum, char * name,
                    const int ndim);
        void decref(void);
        void xdecref(void);
        T& operator()(uint ii);
        T& operator()(uint ii, uint jj);
        T& operator()(uint ii, uint jj, uint kk);
        const uint rows();
        const uint cols();
};

template<class T>
inline void np_wrap<T>::import(const py_ptr to_convert, const int typenum,
                   char * nname, const int edim = 0)
{
    np_ptr tmp;
    if ((tmp = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, NPY_ARRAY_IN_ARRAY))
         == nullptr) {
        print("AAA");
        throw std::runtime_error("");
    }
    
    int array_ndim = PyArray_NDIM(tmp);
    
    if (edim > 0) {
        if (array_ndim != edim) {
            PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                           "expected to be %d-dimensional",
                                            nname, array_ndim, edim);
            throw std::runtime_error("");
        }
    }
    
    ndim = array_ndim;
    shape = PyArray_DIMS(tmp);
    strides = new npy_intp[array_ndim];
    std::memcpy(strides, PyArray_STRIDES(tmp), array_ndim * sizeof(npy_intp));
    
    for(uint ii = 0; ii < array_ndim; ++ii)\
        strides[ii] /= sizeof(*(data));
    
    data = reinterpret_cast<T*>(PyArray_BYTES(tmp));
    np_array = tmp;
    name = nname;
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
inline T& np_wrap<T>::operator()(uint ii)
{
    return data[ii * strides[0]];
}

template<class T>
inline T& np_wrap<T>::operator()(uint ii, uint jj)
{
    return data[  ii * strides[0] + jj * strides[1]];
}

template<class T>
inline T& np_wrap<T>::operator()(uint ii, uint jj, uint kk)
{
    return data[  ii * strides[0] + jj * strides[1] + kk * strides[2]];
}

template<class T>
inline void np_wrap<T>::decref()
{
    Py_DECREF(np_array);
    delete strides;
}

template<class T>
inline void np_wrap<T>::xdecref()
{
    Py_XDECREF(np_array);
    
    if (strides != nullptr)
        delete strides;
}

#if 0

template<class T>
static np_wrap<T> ar_empty(uint ndim, npy_inpt * shape, int typenum,
                           int is_fortran = 0)
{
    np_ptr tmp;
    if (
    
    
}

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


#define ar_empty(ar_struct, edim, shape, typenum)\
do {\
    (ar_struct)

#define ar_array(ar_struct) (ar_struct).np_array

#define ar_ndim(ar_struct) (ar_struct).ndim
#define ar_stride(ar_struct, dim) (uint)(ar_struct).strides[dim]
#define ar_dim(ar_struct, dim) (uint) (ar_struct).shape[dim]

#define ar_rows(ar_struct) ar_dim((ar_struct), 0)
#define ar_cols(ar_struct) ar_dim((ar_struct), 1)

#define ar_data(ar_struct) (ar_struct).data

#define ar_elem1(ar_struct, ii) (ar_struct).data[(ii) * (ar_struct).strides[0]]

#define ar_elem2(ar_struct, ii, jj) (ar_struct).data[\
                                               (ii) * (ar_struct).strides[0]\
                                             + (jj) * (ar_struct).strides[1]]

#define ar_elem3(ar_struct, ii, jj, kk) (ar_struct).data[\
                                                   (ii) * (ar_struct).strides[0]\
                                                 + (jj) * (ar_struct).strides[1]\
                                                 + (kk) * (ar_struct).strides[2]]

#define ar_ptr1(ar_struct, ii) (ar_struct).data + (ii) * (ar_struct).strides[0]

#define ar_ptr2(ar_struct, ii, jj) (ar_struct).data + (ii) * (ar_struct).strides[0] \
                                            + (jj) * (ar_struct).strides[1]

#define ar_ptr3(ar_struct, ii, jj, kk) (ar_struct).data + (ii) * (ar_struct).strides[0]\
                                                 + (jj) * (ar_struct).strides[1]\
                                                 + (kk) * (ar_struct).strides[2]

#define ar_check_matrix(ar_struct, rows, cols)\
do {\
    if (rows != (ar_struct).shape[0]) {\
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of rows=%d "\
                                       "(expected %d)", (ar_struct).name, rows, (ar_struct).shape[0]);
        goto fail;\
    }\

    if (cols != (ar_struct).shape[1]) {\
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of cols=%d "\
                                       "(expected %d)", (ar_struct).name, cols, (ar_struct).shape[1]);
        goto fail;\
    }\
} while(0)

#define ar_check_rows(ar_struct, rows, name)\
do {\
    if (rows != (ar_struct).shape[0]) {\
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of rows=%d "\
                                       "(expected %d)", (ar_struct).name, rows, (ar_struct).shape[0]);
        goto fail;\
    }\
} while (0)

#define ar_check_cols(ar_struct, cols, name)\
do {\
    if (cols != (ar_struct).shape[1]) {\
        PyErr_Format(PyExc_ValueError, "Array %s has wrong number of cols=%d "\
                                       "(expected %d)", (ar_struct).name, cols, (ar_struct).shape[1]);
        goto fail;\
    }\
} while (0)

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



#define ar_decref(ar_struct)\
do {\
    Py_DECREF((ar_struct).np_array);\
    free((ar_struct).strides);\
} while(0)

#define ar_xdecref(ar_struct)\
do {\
    Py_XDECREF((ar_struct).np_array);\
    if ((ar_struct).strides != NULL)\
        free((ar_struct).strides);\
} while(0)

static int _np_convert_array_check(np_ptr * array, const py_ptr to_convert,
                                const int typenum, const int requirements,
                                const int ndim, const char * name)
{
    if ((*array = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, requirements))
         == NULL)
         return 1;

    int array_ndim = PyArray_NDIM(*array);
    
    if (array_ndim != ndim) {
        PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                       "expected to be %d-dimensional", name,
                                        array_ndim, ndim);
        return 1;
    }
    
    return 0;
}

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

#endif

/****************************
 * Numpy convenience macros *
 ****************************/

#define np_gptr1(obj, ii) PyArray_GETPTR1((obj), (ii))
#define np_gptr2(obj, ii, jj) PyArray_GETPTR2((obj), (ii), (jj))

#define np_dim(obj, idx) (uint) PyArray_DIM((obj), (idx))
#define np_ndim(obj) (uint) PyArray_NDIM((obj))

#define np_delem1(obj, ii) *((npy_double *) PyArray_GETPTR1((obj), (ii), (jj)))
#define np_delem2(obj, ii, jj) *((npy_double *) PyArray_GETPTR2((obj), (ii), (jj)))

#define np_belem1(obj, ii) *((npy_bool *) PyArray_GETPTR1((obj), (ii)))
#define np_belem2(obj, ii, jj) *((npy_bool *) PyArray_GETPTR2((obj), (ii), (jj)))

#define np_rows(obj) np_dim((obj), 0)
#define np_cols(obj) np_dim((obj), 1)

#define np_data(obj) PyArray_DATA((obj))

/* based on
 * https://github.com/sniemi/SamPy/blob/master/sandbox/src1/TCSE3-3rd-examples
 * /src/C/NumPy_macros.h */


// Import a numpy array

#define np_import(array_out, array_to_convert, typenum, requirements, name)\
do {\
    if ((array_out) = (np_ptr) PyArray_FROM_OTF(to_convert, typenum,\
                                                requirements) == NULL) {\
        PyErr_Format(PyExc_ValueError, "Failed to import array %s",\
                     (name));\
        goto fail;\
    }\
} while(0)


// Import a numpy array and check the number of its dimensions.

#define np_import_check(array_out, array_to_convert, typenum, requirements,\
                        ndim, name)\
do {\
    if (_np_convert_array_check((&(array_out)), (array_to_convert), (typenum),\
                           (requirements), (ndim), (name))) {\
        goto fail;\
    }\
} while(0)

#define np_import_check_double_in(array_out, array_to_convert, ndim, name)\
        np_import_check((array_out), (array_to_convert), NPY_DOUBLE,\
                        NPY_ARRAY_IN_ARRAY, (ndim), (name))

// Check whether a matrix has the adequate number of rows and cols.

#define np_check_matrix(array, rows, cols, name)\
do {\
    if (_np_check_matrix((array), (rows), (cols), (name)))\
        goto fail;\
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

// Check if a numpy array has the correct number of dimensions.

#define np_check_ndim(array, ndim, name)\
do {\
    if (_np_check_ndim((array), (ndim), (name)))\
        goto fail;\
} while(0)


/* Check whether the number of elements for a given dimension
 * in a numpy array is adequate. */

#define np_check_dim(array, dim, expected_length, name)\
do {\
    if (_np_check_dim((array), (dim), (expected_length), (name)))\
        goto fail;\
} while(0)


// Check the number of rows or cols in a matrix.

#define np_check_rows(obj, expected, name)\
        np_check_dim((obj), 0, (expected), (name))

#define np_check_cols(obj, expected, name)\
        np_check_dim((obj), 1, (expected), (name))

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


// turn s into string "s"
#define QUOTE(s) # s


/***********************************************************
 * Python 2/3 compatible module initialization biolerplate *
 ***********************************************************/


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
    static  PyMethodDef module_name ## _methods[] = {\
    __VA_ARGS__,\
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},\
    {NULL, NULL, 0, NULL}\
    };\

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
