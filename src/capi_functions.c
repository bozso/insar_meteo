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

#include "capi_macros.h"
#include "params_types.h"

int _convert_array_check(np_ptr array, const py_ptr to_convert, const int typenum,
                         const int requirements, const int ndim,
                         const char * name)
{
    if ((array = (np_ptr) PyArray_FROM_OTF(to_convert, typenum, requirements))
         == NULL)
         return 1;
    
    int array_ndim = PyArray_NDIM(array);
    
    if (array_ndim != ndim) {
        PyErr_Format(PyExc_ValueError, "Array %s is %d-dimensional, but "
                                       "expected to be %d-dimensional", name,
                                        array_ndim, ndim);
        return 1;
    }
    
    return 0;
}

int _check_matrix(const np_ptr array, const int rows, const int cols,
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

int _check_ndim(const np_ptr array, const int ndim, const char * name)
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

int _check_dim(const np_ptr array, const int dim, const int expected_length,
               const char * name)
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
