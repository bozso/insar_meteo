#include <stdio.h>
#include "matrix.h"

extern inline void * get_element(matrix * mtx, uint row, uint col);

matrix * allocate_matrix(uint rows, uint cols, size_t elem_size, char * file,
                         int line, char * matrix_name)
{
    matrix * mtx = (matrix *) malloc(sizeof(matrix));
    
    mtx->rows = rows;
    mtx->cols = cols;
    mtx->data = malloc(rows * cols * elem_size);
    
    if (mtx->data == NULL) {
        fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of matrix %s failed!\n",
                        file, line, matrix_name);
        free(mtx);
        return NULL;
    }
    else
        return mtx;
}

void matrix_safe_free(matrix * mtx)
{
    if (mtx != NULL) {
        free(mtx->data);
        free(mtx);
        mtx = NULL;
    }
}

void matrix_free(matrix * mtx)
{
        free(mtx->data);
        free(mtx);
}
