#include "nparray.hh"
#include "utils.hh"

#define handle_shape \
npy_intp _shape[num];\
va_list vl;\
va_start(vl, num);\
for(size_t ii = 0; ii < num; ++ii)\
    _shape[ii] = va_arg(vl, size_t);\
va_end(vl)


static void setup_array(nparray *arr, PyArrayObject *_array, size_t const edim);

size_t calc_size(size_t num_array, size_t numdim)
{
    return num_array * numdim * 2 * sizeof(size_t);
}


nparray::nparray(int const typenum, size_t const ndim, PyObject *obj)
{
    npobj = (PyArrayObject*) PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_IN_ARRAY);
    
    if (npobj == NULL and not PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "Failed to convert numpy nparray!");
        return;
    }
    
    setup_array(this, npobj, ndim);
}


nparray::nparray(int const typenum, void *data, size_t num, ...)
{
    handle_shape;
    npobj = (PyArrayObject*) PyArray_SimpleNewFromData(num, _shape, typenum, data);
    
    if (npobj == NULL and not PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return;
    }
    
    setup_array(this, npobj, 0);
}


nparray::nparray(int const typenum, newtype const newt, char const layout,
                 size_t num, ...)
{
    handle_shape;
    int fortran;
    
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
    
    
    switch (newt) {
        case empty:
            npobj = (PyArrayObject*) PyArray_EMPTY(num, _shape, typenum, fortran);
            break;
        case zeros:
            npobj = (PyArrayObject*) PyArray_ZEROS(num, _shape, typenum, fortran);
            break;
        default:
            // raise Exception
            return;
    }
    
    if (npobj == NULL and not PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        return;
    }
    
    setup_array(this, npobj, 0);
}


nparray::~nparray()
{
    decref();
    //if (strides)
        //PyMem_Del(strides);
    
    strides = shape = NULL;
    
    if (_decref)
        Py_CLEAR(npobj);
}


PyArrayObject* nparray::ret()
{
    _decref = false;
    return npobj;
}


bool nparray::check_rows(size_t const rows) const
{
    if (shape[0] != rows) {
        PyErr_Format(PyExc_TypeError, "Expected array to have rows %u but got "
                     "array with rows %u.", rows, shape[0]);
        return true;
    }
    return false;
}


bool nparray::check_cols(size_t const cols) const
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


static void setup_array(nparray *arr, PyArrayObject *_array, size_t const edim)
{
    size_t _ndim = size_t(PyArray_NDIM(_array));
    
    if (edim and (edim != _ndim) and not PyErr_Occurred()) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return;
    }
    
    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    incref();
    arr->strides = palloc(size_t, 2 * _ndim);
    arr->shape = arr->strides + _ndim;
    
    npy_intp * strides = PyArray_STRIDES(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->strides[ii] = size_t(double(strides[ii]) / elemsize);

 
    npy_intp *shape = PyArray_DIMS(_array);
    
    for(size_t ii = _ndim; ii--; )
        arr->shape[ii] = size_t(shape[ii]);
    
    arr->_decref = true;
}
