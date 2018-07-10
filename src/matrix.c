#include <stdio.h>
#include "matrix.h"

matrix * allocate_matrix(uint rows, uint cols, size_t elem_size)
{
    matrix * mtx = (matrix *) malloc(sizeof(matrix));
    
    mtx->rows = rows;
    mtx->cols = cols;
    mtx->data = malloc(rows * cols * elem_size);
    
    return mtx;
}
