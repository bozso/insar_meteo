#include "nparray.hh"

static bool setup_array(nparray* arr, PyArrayObject *_array, size_t const edim = 0)
{
    int _ndim = size_t(PyArray_NDIM(_array));
    
    if (edim and edim != _ndim) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return true;
        
    }
    
    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    size_t *tmp = PyMem_New(size_t, 2 * _ndim)
    
    if (tmp == NULL) {
        // raise Exception
        return true;
    }
    
    arr->strides = tmp;
    arr->shape = tmp + _ndim;
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->strides[ii] = size_t(double(strides[ii]) / elemsize);

 
    npy_intp *shape = PyArray_DIMS(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->shape[ii] = size_t(shape[ii]);
    
    arr->decref = true;
    return false;
}


bool nparray::from_data(Pool& pool, npy_intp * dims, void *data)
{
    if ((npobj = (PyArrayObject*) PyArray_SimpleNewFromData(ndim, dims,
                      typenum, data)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(this, npobj, 0);
}


bool nparray::import(Pool& pool, PyObject *obj)
{
    if (obj != NULL)
        pyobj = obj;
    
    if ((npobj = (PyArrayObject*) PyArray_FROM_OTF(pyobj, typenum,
                                           NPY_ARRAY_IN_ARRAY)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to convert numpy nparray!");
        return true;
    }
    
    return setup_array(this, npobj, ndim);
}


bool nparray::empty(Pool& pool, npy_intp* dims, int const fortran)
{
    if ((npobj = (PyArrayObject*) PyArray_EMPTY(ndim, dims,
                      typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(pool, this, npobj, 0);
}


bool nparray::zeros(Pool& pool, npy_intp * dims, int const fortran)
{
    if ((npobj = (PyArrayObject*) PyArray_ZEROS(ndim, dims, typenum,
                                                fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(this, npobj, 0);
}


void * nparray::data() const {
    return PyArray_DATA(arr.npobj);
}


PyArrayObject* nparray::ret()
{
    decref = false;
    return npobj;
}


bool nparray::check_rows(npy_intp const rows) const
{
    if (shape[0] != rows) {
        PyErr_Format(PyExc_TypeError, "Expected array to have rows %u but got "
                     "array with rows %u.", rows, shape[0]);
        return true;
    }
    return false;
}


bool nparray::check_cols(npy_intp const cols) const
{
    if (shape[1] != cols) {
        PyErr_Format(PyExc_TypeError, "Expected array to have cols %u but got "
                     "array with cols %u.", cols, shape[1]);
        return true;
    }
    return false;
}


bool nparray::is_f_cont() const {
    return PyArray_IS_F_CONTIGUOUS(npobj);
}
