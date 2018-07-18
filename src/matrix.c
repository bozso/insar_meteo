/* INMET
 * Copyright (C) 2018  MTA CSFK GGI
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

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "main_functions.h"
#include "matrix.h"

matrix * mtx_allocate(const size_t rows, const size_t cols,
                      const data_type dtype, const char * file, const int line,
                      const char * matrix_name)
{
    matrix * mtx;
    
    if ((mtx = (matrix *) malloc(sizeof(matrix))) == NULL)
        goto fail;
    
    switch(dtype) {
        case data_complex_long_double:
            mtx->matrix_un.matrix_complex_long_double = 
            gsl_matrix_complex_long_double_alloc(rows, cols);
            break;
        case data_complex_double:
            mtx->matrix_un.matrix_complex_double = 
            gsl_matrix_complex_alloc(rows, cols);
            break;
        case data_complex_float:
            mtx->matrix_un.matrix_complex_float = 
            gsl_matrix_complex_float_alloc(rows, cols);
            break;

        case data_long_double:
            mtx->matrix_un.matrix_long_double = 
            gsl_matrix_long_double_alloc(rows, cols);
            break;
        case data_double:
            mtx->matrix_un.matrix_double = 
            gsl_matrix_alloc(rows, cols);
            break;
        case data_float:
            mtx->matrix_un.matrix_float = 
            gsl_matrix_float_alloc(rows, cols);
            break;

        case data_ulong:
            mtx->matrix_un.matrix_ulong = 
            gsl_matrix_ulong_alloc(rows, cols);
            break;
        case data_long:
            mtx->matrix_un.matrix_long = 
            gsl_matrix_long_alloc(rows, cols);
            break;

        case data_ushort:
            mtx->matrix_un.matrix_ushort = 
            gsl_matrix_ushort_alloc(rows, cols);
            break;
        case data_short:
            mtx->matrix_un.matrix_short = 
            gsl_matrix_short_alloc(rows, cols);
            break;

        case data_uchar:
            mtx->matrix_un.matrix_uchar = 
            gsl_matrix_uchar_alloc(rows, cols);
            break;
        case data_char:
            mtx->matrix_un.matrix_char = 
            gsl_matrix_char_alloc(rows, cols);
            break;
        default:
            fprintf(stderr, "Unrecognized data_type!");
            goto fail;
    }
    
    mtx->dtype = dtype;
    
    return mtx;
    
fail:
    fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of matrix %s failed!\n",
                    file, line, matrix_name);
    aux_free(mtx);
    return NULL;
}

vector * vec_allocate(const size_t elems,const data_type dtype,
                      const char * file, const int line,
                      const char * vector_name)
{
    vector * vec;
    
    if ((vec = (vector *) malloc(sizeof(vector))) == NULL)
        goto fail;
    
    switch(dtype) {
        case data_complex_long_double:
            vec->vector_un.vector_complex_long_double = 
            gsl_vector_complex_long_double_alloc(elems);
            break;
        case data_complex_double:
            vec->vector_un.vector_complex_double = 
            gsl_vector_complex_alloc(elems);
            break;
        case data_complex_float:
            vec->vector_un.vector_complex_float = 
            gsl_vector_complex_float_alloc(elems);
            break;

        case data_long_double:
            vec->vector_un.vector_long_double = 
            gsl_vector_long_double_alloc(elems);
            break;
        case data_double:
            vec->vector_un.vector_double = 
            gsl_vector_alloc(elems);
            break;
        case data_float:
            vec->vector_un.vector_float = 
            gsl_vector_float_alloc(elems);
            break;

        case data_ulong:
            vec->vector_un.vector_ulong = 
            gsl_vector_ulong_alloc(elems);
            break;
        case data_long:
            vec->vector_un.vector_long = 
            gsl_vector_long_alloc(elems);
            break;

        case data_ushort:
            vec->vector_un.vector_ushort = 
            gsl_vector_ushort_alloc(elems);
            break;
        case data_short:
            vec->vector_un.vector_short = 
            gsl_vector_short_alloc(elems);
            break;

        case data_uchar:
            vec->vector_un.vector_uchar = 
            gsl_vector_uchar_alloc(elems);
            break;
        case data_char:
            vec->vector_un.vector_char = 
            gsl_vector_char_alloc(elems);
            break;
        default:
            fprintf(stderr, "Unrecognized data_type!");
            goto fail;
    }
    
    vec->dtype = dtype;
    
    return vec;
    
fail:
    fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of vector %s failed!\n",
                    file, line, vector_name);
    aux_free(vec);
    return NULL;
}

#if 0

extern inline char * get_element(const matrix * mtx, const uint row, const uint col);

matrix * mtx_allocate(const uint rows, const uint cols, const size_t elem_size,
                      const data_type type, const char * file, const int line,
                      const char * matrix_name)
{
    matrix * mtx = (matrix *) malloc(sizeof(matrix));
    
    if (mtx == NULL)
        goto fail;
    
    mtx->rows = rows;
    mtx->cols = cols;
    mtx->type = type;
    mtx->elem_size = elem_size;
    mtx->data = (char *) malloc(rows * cols * elem_size);
    
    if (mtx->data == NULL)
        goto fail;
    
    return mtx;

fail:
    fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of matrix %s failed!\n",
                    file, line, matrix_name);
    aux_free(mtx);
    return NULL;
}

void mtx_safe_free(matrix * mtx)
{
    if (mtx != NULL) {
        free(mtx->data);
        free(mtx);
        mtx = NULL;
    }
}

void mtx_free(matrix * mtx)
{
        free(mtx->data);
        free(mtx);
}

#endif
