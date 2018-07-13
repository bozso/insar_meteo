#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "main_functions.h"
#include "matrix.h"

extern inline char * get_element(matrix * mtx, uint row, uint col);

matrix * mtx_allocate(uint rows, uint cols, size_t elem_size, data_type type,
                      char * file, int line, char * matrix_name)
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
