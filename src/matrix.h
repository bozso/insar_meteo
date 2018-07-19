/* Copyright (C) 2018  MTA CSFK GGI
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

#ifndef MATRIX_H
#define MATRIX_H

typedef enum data_type_t {
    data_complex_long_double,
    data_complex_double,
    data_complex_float,
    
    data_long_double,
    data_double,
    data_float,
    
    data_ulong,
    data_long,

    data_uint,
    data_int,
    
    data_ushort,
    data_short,
    
    data_uchar,
    data_char,
    
    data_other
} data_type;

typedef enum io_type_t {
    io_ascii,
    io_binary
} io_type;

typedef struct matrix_t {
    size_t rows, cols, elem_size;
    data_type type;
    char * data;
} matrix;


matrix * mtx_allocate(const size_t rows, const size_t cols, const size_t elem_size,
                      const data_type type, const char * file, const int line,
                      const char * matrix_name);

void mtx_safe_free(matrix * mtx);
void mtx_free(matrix * mtx);

inline char * get_element(const matrix * mtx, const uint row, const uint col)
{
    return mtx->data + (col + row * mtx->cols) * mtx->elem_size;
}

#define mtx_init(mtx, rows, cols, type, dtype)\
({\
    if (((mtx) = mtx_allocate(rows, cols, sizeof(type), dtype,\
            __FILE__, __LINE__, #mtx)) == NULL)\
        goto fail;\
})

#define mtx_double(mtx, rows, cols) mtx_init(mtx, rows, cols, double, data_double)
#define mtx_float(mtx, rows, cols) mtx_init(mtx, rows, cols, float, data_float)
#define mtx_int(mtx, rows, cols) mtx_init(mtx, rows, cols, int, data_int)

#define mtx_other(mtx, rows, cols, type) mtx_init(mtx, rows, cols, type, data_other)

#define emtx(mtx, row, col, type)\
    *((type *) get_element(mtx, row, col))

#define dmtx(mtx, row, col)\
    *((double *) get_element(mtx, row, col))

#define fmtx(mtx, row, col)\
    *((float *) get_element(mtx, row, col))

#define imtx(mtx, row, col)\
    *((int *) get_element(mtx, row, col))

#define mtxe(array, row, col, cols) array[col + row * cols]

#if 0

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef union gsl_ge_matrix_un_t {
    gsl_matrix_complex_long_double  * matrix_complex_long_double;
    gsl_matrix_complex              * matrix_complex_double;
    gsl_matrix_complex_float        * matrix_complex_float;
    
    gsl_matrix_long_double  * matrix_long_double;
    gsl_matrix              * matrix_double;
    gsl_matrix_float        * matrix_float;
    
    gsl_matrix_ulong    * matrix_ulong;
    gsl_matrix_long     * matrix_long;
    
    gsl_matrix_uint * matrix_uint;
    gsl_matrix_int  * matrix_int;
    
    gsl_matrix_ushort   * matrix_ushort;
    gsl_matrix_short    * matrix_short;
    
    gsl_matrix_uchar    * matrix_uchar;
    gsl_matrix_char     * matrix_char;
} gsl_ge_matrix_un;

typedef struct matrix_t {
    data_type dtype;
    size_t nrow, ncol;
    gsl_ge_matrix_un matrix_un;
} matrix;

typedef union gsl_ge_vector_un_t {
    gsl_vector_complex_long_double  * vector_complex_long_double;
    gsl_vector_complex              * vector_complex_double;
    gsl_vector_complex_float        * vector_complex_float;
    
    gsl_vector_long_double  * vector_long_double;
    gsl_vector              * vector_double;
    gsl_vector_float        * vector_float;
    
    gsl_vector_ulong    * vector_ulong;
    gsl_vector_long     * vector_long;
    
    gsl_vector_uint * vector_uint;
    gsl_vector_int  * vector_int;
    
    gsl_vector_ushort   * vector_ushort;
    gsl_vector_short    * vector_short;
    
    gsl_vector_uchar    * vector_uchar;
    gsl_vector_char     * vector_char;
} gsl_ge_vector_un;

typedef struct vector_t {
    data_type dtype;
    size_t nelem;
    gsl_ge_vector_un vector_un;
} vector;

#define mtx_init(mtx, rows, cols, dtype)\
({\
    if (((mtx) = mtx_allocate(rows, cols, dtype, __FILE__, __LINE__, #mtx)) \
                == NULL)\
        error = err_alloc;\
        goto fail;\
})

#define vec_init(vec, elems, dtype)\
({\
    if (((vec) = vec_allocate(elems, dtype, __FILE__, __LINE__, #vec)) \
                == NULL)\
        error = err_alloc;\
        goto fail;\
})

matrix * mtx_allocate(const size_t nrow, const size_t ncol,
                      const data_type dtype, const char * file, const int line,
                      const char * matrix_name);

vector * vec_allocate(const size_t nelem,const data_type dtype,
                      const char * file, const int line,
                      const char * vector_name);

void mtx_free(matrix * mtx);
void vec_free(vector * vec);

#define mtx_safe_free(mtx)\
({\
    if (mtx != NULL) {\
        mtx_free(mtx);\
        mtx = NULL;\
    }\
})

#define vec_safe_free(vec)\
({\
    if (vec != NULL) {\
        vec_free(vec);\
        vec = NULL;\
    }\
})

#endif

#endif
