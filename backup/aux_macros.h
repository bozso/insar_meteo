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

#define OR ||
#define AND &&

/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/**********************************************
 * Documentation and argument number checking *
 **********************************************/

#define print_doc(fun_name) printf("%s", fun_name ## __doc__)

#define aux_checkarg(num, doc)\
({\
    if (argc != ((num) + min_arg)) {\
        errorln("\n Required number of arguments is %d, current number of arguments: %d!",\
                 (num), argc - min_arg);\
        printf((doc));\
        error = err_arg;\
        goto fail;\
    }\
})

/*****************************
 * Switching between modules *
 *****************************/

#define str_isequal(string1, string2) (strcmp((string1), (string2)) == 0)
#define module_select(string) str_isequal(argv[1], string)

/*********************
 * Allocation macros *
 *********************/

#define aux_malloc(ptr, type, num)\
({\
    if (NULL == ((ptr) = (type *) malloc(sizeof(type) * (num))))\
    {\
        fprintf(stderr, "FILE: %s, LINE: %d :: Malloc of %s failed",\
                __FILE__, __LINE__, #ptr);\
        error = err_alloc;\
        goto fail;\
    }\
})

#define aux_free(pointer)\
({\
    if ((pointer) != NULL)\
    {\
        free((pointer));\
        pointer = NULL;\
    }\
})

#define aux_alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define aux_realloc(pointer, type, size) (pointer) = (type *) \
                                     realloc(pointer, (size) * sizeof(type))	

/**************************
 * GSL convenience macros *
 **************************/

#define Mset(mtx, ii, jj, data) gsl_matrix_set((mtx), (ii), (jj), (data))
#define Vset(vector, ii, data) gsl_vector_set((vector), (ii), (data))

#define Mget(mtx, ii, jj) gsl_matrix_get((mtx), (ii), (jj))
#define Vget(vector, ii) gsl_vector_get((vector), (ii))

#define Mptr(mtx, ii, jj) gsl_matrix_ptr((mtx), (ii), (jj))
#define Vptr(vector, ii) gsl_vector_ptr((vector), (ii))

/*************
 * IO macros *
 *************/

#define error(string) fprintf(stderr, string)
#define errorln(format, ...) fprintf(stderr, format "\n", __VA_ARGS__)

#define println(format, ...) printf(format "\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

#define aux_open(file, path, mode)\
({\
    if (((file) = fopen((path), (mode))) == NULL)\
    {\
        errorln("FILE: %s, LINE: %d :: Failed to open file %s!", __FILE__, __LINE__, path);\
        perror("Error");\
        error = errno;\
        goto fail;\
    }\
})

#define aux_close(file)\
({\
    if ((file) != NULL) {\
        fclose((file));\
        (file) = NULL;\
    }\
})


#define aux_fscanf(file, format, ...)\
({\
    if (fscanf(file, format, __VA_ARGS__) <= 0) {\
        error = err_io;\
        goto fail;\
    }\
})


#if 0

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

#endif
