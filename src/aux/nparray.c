#include "nparray.h"

static void setup_array(nparray *arr, PyArrayObject *array, size_t const edim);

nparray * nparray_init(int const typenum, size_t const ndim, PyObject* obj)
{
    nparray *arr;
    if ((arr = Mem_New(nparray, 1)) == NULL) {
        return NULL;
    }
    
}


static void dtor_(void *obj)
{
    PyXDECREF(((nparray *)obj)->npobj);
}


static bool setup_array(nparray *arr, PyArrayObject *array, size_t const edim)
{
    size_t _ndim = (size_t)PyArray_NDIM(array);
    
    if (edim and (edim != _ndim)) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return false;
    }
    
    int elemsize = (int) PyArray_ITEMSIZE(array);
    
    size_t *tmp = PyMem_New(size_t, 2 * _ndim);
    
    if (tmp == NULL) {
        PyErr_Format(PyExc_MemoryError, "Could not allocate memory for numpy "
                     "array shapes and strides.");
        return false;
    }
    
    arr->strides = tmp;
    arr->shape = arr->strides + _ndim;
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->strides[ii] = (size_t) ((double) strides[ii] / elemsize);

 
    npy_intp *shape = PyArray_DIMS(array);
    
    for(size_t ii = _ndim; ii--; )
        arr->shape[ii] = (size_t) shape[ii];
    
    arr->dtor_ = &dtor_;
    
    return false;
}
