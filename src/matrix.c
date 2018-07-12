#include <stdio.h>
#include "matrix.h"

extern inline void * get_element(matrix * mtx, uint row, uint col);

matrix * allocate_matrix(uint rows, uint cols, size_t elem_size, mtx_type type,
                         char * file, int line, char * matrix_name)
{
    matrix * mtx = (matrix *) malloc(sizeof(matrix));
    
    if (mtx == NULL)
        goto fail;
    
    mtx->rows = rows;
    mtx->cols = cols;
    mtx->type = type;
    mtx->data = malloc(rows * cols * elem_size);
    
    if (mtx->data == NULL)
        goto fail;
    
    return mtx;

fail:
    fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of gmatrix %s failed!\n",
                    file, line, matrix_name);
    aux_free(mtx);
    return NULL;

}

#if 0

matrix * allocate_matrix(uint rows, uint cols, mtx_type type, char * file,
                         int line, char * matrix_name)
{
    matrix * mtx = (matrix *) malloc(sizeof(matrix));

    if (mtx == NULL)
        goto fail;

    mtx->rows = rows;
    mtx->cols = cols;
    mtx->type = type;
    
    switch(type) {
        case mtx_double:
            aux_allocate(mtx->ddata, rows * cols, double);
        case mtx_float:
            aux_allocate(mtx->fdata, rows * cols, float);
        case mtx_int:
            aux_allocate(mtx->idata, rows * cols, int);
        default:
            errorln("Unrecognized matrix type: %u", type);
            goto fail;
        
    }
    
    return mtx;
fail:
    fprintf(stderr, "FILE: %s, LINE: %d :: Allocation of matrix %s failed!\n",
                    file, line, matrix_name);
    aux_free(mtx);
    return NULL;
}

#endif


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
