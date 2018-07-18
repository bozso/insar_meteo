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
