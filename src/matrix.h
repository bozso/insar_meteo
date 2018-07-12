#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include "main_functions.h"

typedef enum mtx_type_t {
    mtx_double,
    mtx_float,
    mtx_int
} mtx_type;

typedef struct matrix_t {
    uint rows, cols;
    size_t elem_size;
    mtx_type type;
    void * data;
} matrix;


matrix * allocate_matrix(uint rows, uint cols, size_t elem_size, mtx_type type,
                         char * file, int line, char * matrix_name);

void matrix_safe_free(matrix * mtx);
void matrix_free(matrix * mtx);

inline void * get_element(matrix * mtx, uint row, uint col)
{
    return mtx->data + (col + row * mtx->cols) * mtx->elem_size;
}

#define init_matrix(mtx, rows, cols, var_type, type)\
({\
    if (((mtx) = allocate_matrix(rows, cols, var_type, sizeof(type),\
            __FILE__, __LINE__, #mtx)) == NULL)\
        goto fail;\
})

#define matrix_double(mtx, rows, cols) init_matrix(mtx, rows, cols, mtx_double, double)
#define matrix_float(mtx, rows, cols) init_matrix(mtx, rows, cols, mtx_float, float)
#define matrix_int(mtx, rows, cols) init_matrix(mtx, rows, cols, mtx_int, int)

#define melem(mtx, row, col, type)\
    *((type *) get_element(mtx, row, col))

#define delem(mtx, row, col)\
    *((double *) get_element(mtx, row, col))

#define felem(mtx, row, col)\
    *((float *) get_element(mtx, row, col))

#define ielem(mtx, row, col)\
    *((int *) get_element(mtx, row, col))

#endif
