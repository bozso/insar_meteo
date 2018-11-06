#include "nparray.hh"

static bool const setup_array(nparray& arr, PyArrayObject *_array, size_t const edim = 0)
{
    int _ndim = size_t(PyArray_NDIM(_array));
    
    if (edim and edim != _ndim) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return true;
        
    }
    
    arr.shape = PyArray_DIMS(_array);

    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    if ((arr.strides = PyMem_New(size_t, _ndim)) == NULL) {
        // TODO raise exception
        return true;
    }

    if ((arr.shape = PyMem_New(size_t, _ndim)) == NULL) {
        // TODO raise exception
        return true;
    }
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr.strides[ii] = size_t(double(strides[ii]) / elemsize);
    
    return false;
}


bool const from_data(nparray& arr, npy_intp * dims, void *data)
{
    if ((arr.npobj = (PyArrayObject*) PyArray_SimpleNewFromData(arr.ndim, dims,
                      arr.typenum, data)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj, 0);
}


bool const import(nparray& arr, PyObject *_obj = NULL)
{
    if (_obj != NULL)
        pyobj = _obj;
    
    if ((npobj =
         (PyArrayObject*) PyArray_FROM_OTF(pyobj, arr.typenum,
                                           NPY_ARRAY_IN_ARRAY)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to convert numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj, arr.ndim);
}


bool const empty(nparray& arr, npy_intp* dims, int const fortran = 0)
{
    if ((npobj = (PyArrayObject*) PyArray_EMPTY(arr.ndim, dims,
                      arr.typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj, 0);
}


bool const zeros(nparray& arr, npy_intp * dims, int const fortran = 0)
{
    if ((npobj = (PyArrayObject*) PyArray_ZEROS(arr.ndim, dims,
                      arr.typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return true;
    }
    
    return setup_array(arr, npobj, 0);
}

size_t const rows(nparray const& arr) {
    return arr.shape[0];
}

size_t const cols(nparray const& arr) {
    return arr.shape[1];
}

PyArrayObject* ret(nparray& arr)
{
    arr.decref = false;
    return arr.npobj;
}


bool const check_rows(nparray const& arr, npy_intp const rows)
{
    if (arr.shape[0] != rows) {
        PyErr_Format(PyExc_TypeError, "Expected array to have rows %u but got "
                     "array with rows %u.", rows, shape[0]);
        return true;
    }
    return false;
}


bool const check_cols(nparray const& arr, npy_intp const cols)
{
    if (arr.shape[1] != cols) {
        PyErr_Format(PyExc_TypeError, "Expected array to have cols %u but got "
                     "array with cols %u.", cols, shape[1]);
        return true;
    }
    return false;
}


bool const is_f_cont(nparray const& arr) {
    return PyArray_IS_F_CONTIGUOUS(arr.npobj);
}
