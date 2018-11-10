#include "nparray.hh"

#define handle_shape \
npy_intp *_shape;\
\
if ((_shape = new npy_intp[num]) == NULL)\
    return true;\
\
va_list vl;\
va_start(vl, num);\
for(size_t ii = 0; ii < num; ++ii)\
    _shape[ii] = va_arg(vl, size_t);\
va_end(vl)


static bool setup_array(nparray *arr, PyArrayObject *_array, size_t const edim)
{
    size_t _ndim = size_t(PyArray_NDIM(_array));
    
    if (edim and (edim != _ndim)) {
        PyErr_Format(PyExc_TypeError, "numpy nparray expected to be %u "
                    "dimensional but we got %u dimensional nparray!",
                    edim, _ndim);
        return true;
        
    }
    
    int elemsize = int(PyArray_ITEMSIZE(_array));
    
    size_t *tmp = new size_t[2 * _ndim];
    
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


bool nparray::import(int const typenum, size_t const ndim, PyObject *obj)
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


bool nparray::from_data(int const typenum, void *data, size_t num, ...)
{
    handle_shape;
    
    if ((npobj = (PyArrayObject*) PyArray_SimpleNewFromData(num, _shape,
                      typenum, data)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        delete[] _shape;
        return true;
    }
    
    delete[] _shape;
    
    return setup_array(this, npobj, 0);
}


bool nparray::empty(int const typenum, int const fortran, size_t num, ...)
{
    handle_shape;
    
    if ((npobj = (PyArrayObject*) PyArray_EMPTY(num, _shape,
                      typenum, fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        delete[] shape;
        return true;
    }
    
    delete[] shape;
    return setup_array(this, npobj, 0);
}


bool nparray::zeros(int const typenum, int const fortran, size_t num, ...)
{
    handle_shape;
    
    if ((npobj = (PyArrayObject*) PyArray_ZEROS(num, _shape, typenum,
                                                fortran)) == NULL) {
        PyErr_Format(PyExc_TypeError, "Failed to create numpy nparray!");
        delete[] _shape;
        return true;
    }
    
    delete[] _shape;
    return setup_array(this, npobj, 0);
}


void * nparray::data() const {
    return PyArray_DATA(npobj);
}


PyArrayObject* nparray::ret()
{
    decref = false;
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
