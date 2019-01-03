#ifndef NPARRAY_HH
#define NPARRAY_HH

#include <stddef.h>
#include <stdbool.h>

#include "common.h"
#include "numpy/arrayobject.h"


extern_begin

struct _nparray {
    int typenum;
    size_t ndim, *shape, *strides;
    PyArrayObject *npobj;
    dtor dtor_;
};

typedef struct _nparray* nparray;

typedef enum newtype {
    empty,
    zeros
} newtype;


nparray from_otf(int const typenum, size_t const ndim, PyObject* obj);
nparray from_of(size_t const ndim, PyObject *obj);
nparray from_data(int const typenum, void *data, size_t ndim, npy_intp *shape);
nparray newarray(int const typenum, newtype const newt, char const layout,
                 size_t ndim, npy_intp *shape);
void * ar_data(nparray const arr);

bool check_rows(nparray arr, size_t const rows);
bool check_cols(nparray arr, size_t const cols);
bool is_f_cont(nparray arr);
bool is_zero_dim(nparray arr);
bool check_scalar(nparray arr);
bool is_python_number(nparray arr);
bool is_not_swapped(nparray arr);
bool is_byte_swapped(nparray arr);
bool can_cast_to(nparray arr, int const totypenum);

typedef enum dtype {
    np_bool = NPY_BOOL,

    np_byte = NPY_BYTE,
    np_ubyte = NPY_UBYTE,
    np_short = NPY_SHORT,
    np_ushort = NPY_USHORT,
    np_int = NPY_INT,
    np_uint = NPY_UINT,
    np_long = NPY_LONG,
    np_ulong = NPY_ULONG,
    np_longlong = NPY_LONGLONG,
    np_ulonglong = NPY_ULONGLONG,

    np_double = NPY_DOUBLE,
    np_longdouble = NPY_LONGDOUBLE,
    np_cfloat = NPY_CFLOAT,
    np_cdouble = NPY_CDOUBLE,
    np_clongdouble = NPY_CLONGDOUBLE,
    np_object = NPY_OBJECT,
    
    np_int8 = NPY_INT8,
    np_int16 = NPY_INT16,
    np_int64 = NPY_INT64,
    np_uint8 = NPY_UINT8,
    np_uint16 = NPY_UINT16,
    np_uint64 = NPY_UINT64,
    
    np_float32 = NPY_FLOAT32,
    np_float64 = NPY_FLOAT64,
    
    np_complex64 = NPY_COMPLEX64,
    np_complex128 = NPY_COMPLEX128
} dtype;


#ifdef m_get_impl


static nparray init_array(PyObject const * arr);
static bool setup_array(nparray arr, size_t const edim);

nparray from_otf(int const typenum, size_t const ndim, PyObject* obj)
{
    
    nparray arr = init_array(PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_IN_ARRAY));

    if (arr == NULL)
        return NULL;
    
    if (setup_array(arr, ndim)) {
        Mem_Free(arr);
        return NULL;
    }
    return arr;
}


nparray from_of(size_t const ndim, PyObject *obj)
{
    nparray arr = init_array(PyArray_FROM_OF(obj, NPY_ARRAY_IN_ARRAY));

    if (arr == NULL)
        return NULL;

    if (setup_array(arr, ndim)) {
        Mem_Free(arr);
        return NULL;
    }

    return arr;
}



nparray from_data(int const typenum, void *data, size_t ndim, npy_intp *shape)
{
    nparray arr = init_array(PyArray_SimpleNewFromData(ndim, shape, typenum, data));

    if (arr == NULL)
        return NULL;
    
    if (setup_array(arr, 0)) {
        Mem_Free(arr);
        return NULL;
    }

    return arr;
}


nparray newarray(int const typenum, newtype const newt, char const layout,
                 size_t ndim, npy_intp *shape)
{
    int fortran = 0;
    
    switch (layout) {
        case 'C':
            fortran = 0;
            break;
        case 'c':
            fortran = 0;
            break;
        case 'F':
            fortran = 1;
            break;
        case 'f':
            fortran = 1;
            break;
    }
    
    PyObject *npobj = NULL;
    
    switch (newt) {
        case empty:
            npobj = PyArray_EMPTY(ndim, shape, typenum, fortran);
            break;
        case zeros:
            npobj = PyArray_ZEROS(ndim, shape, typenum, fortran);
            break;
        default:
            // raise Exception
            return NULL;
    }
    
    nparray arr = init_array(npobj);

    if (arr == NULL)
        return NULL;

    if (setup_array(arr, 0)) {
        Mem_Free(arr);
        return NULL;
    }

    return arr;
}


bool check_rows(nparray arr, size_t const rows)
{
    if (arr->shape[0] != rows) {
        PyErr_Format(PyExc_TypeError, "Expected array to have rows %u but got "
                     "array with rows %u.", rows, arr->shape[0]);
        return true;
    }
    return false;
}


bool check_cols(nparray arr, size_t const cols)
{
    if (arr->shape[1] != cols) {
        PyErr_Format(PyExc_TypeError, "Expected array to have cols %u but got "
                     "array with cols %u.", cols, arr->shape[1]);
        return true;
    }
    return false;
}


bool is_f_cont(nparray arr) { return PyArray_IS_F_CONTIGUOUS(arr->npobj); }

bool is_zero_dim(nparray arr) { return PyArray_IsZeroDim(arr->npobj); }

bool check_scalar(nparray arr) { return PyArray_CheckScalar(arr->npobj); }

bool is_python_number(nparray arr) { return PyArray_IsPythonNumber(arr->npobj); }

bool is_python_scalar(nparray arr) { return PyArray_IsPythonScalar(arr->npobj); }

bool is_not_swapped(nparray arr) { return PyArray_ISNOTSWAPPED(arr->npobj); }

bool is_byte_swapped(nparray arr) { return PyArray_ISBYTESWAPPED(arr->npobj); }

bool can_cast_to(nparray arr, int const totypenum)
{
    return PyArray_CanCastTo(PyArray_DescrFromType(arr->typenum),
                             PyArray_DescrFromType(totypenum));
}


void * ar_data(nparray const arr)
{
    return PyArray_DATA(arr->npobj);
}


static void nparray_dtor(void *obj)
{
    Py_XDECREF(((nparray)obj)->npobj);
}


static nparray init_array(PyObject const * nparr)
{
    if (nparr == NULL)
        return NULL;
    
    nparray arr = Mem_New(struct _nparray, 1);
    
    if (arr == NULL)
        return NULL;

    arr->npobj = (PyArrayObject*) nparr;
    
    return arr;
}

static bool setup_array(nparray arr, size_t const edim)
{
    PyArrayObject *array = arr->npobj;
    size_t _ndim = (size_t)PyArray_NDIM(array);
    
    if (edim and (edim != _ndim)) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return false;
    }
    
    int elemsize = (int) PyArray_ITEMSIZE(array);
    
    size_t *tmp = Mem_New(size_t, 2 * _ndim);
    
    if (tmp == NULL) {
        PyErr_Format(PyExc_MemoryError, "Could not allocate memory for numpy "
                     "array shapes and strides.");
        return false;
    }
    
    arr->strides = tmp;
    arr->shape = arr->strides + _ndim;
    
    npy_intp * strides = PyArray_STRIDES(array);
    
    for(size_t ii = _ndim; ii--; )
        arr->strides[ii] = (size_t) ((double) strides[ii] / elemsize);

 
    npy_intp *shape = PyArray_DIMS(array);
    
    for(size_t ii = _ndim; ii--; )
        arr->shape[ii] = (size_t) shape[ii];
    
    arr->dtor_ = &nparray_dtor;
    
    return false;
}

#endif

extern_end

#endif
