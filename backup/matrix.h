#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include "main_functions.h"

typedef enum mtx_type_t {
    vtype_double,
    vtype_float,
    vtype_int,
    vtype_other
} mtx_type;

typedef struct matrix_t {
    uint rows, cols;
    size_t elem_size;
    mtx_type type;
    void * data;
} matrix;


matrix * mtx_allocate(uint rows, uint cols, size_t elem_size, mtx_type type,
                      char * file, int line, char * matrix_name);

void mtx_safe_free(matrix * mtx);
void mtx_free(matrix * mtx);

inline void * get_element(matrix * mtx, uint row, uint col)
{
    return mtx->data + (col + row * mtx->cols) * mtx->elem_size;
}

#define mtx_init(mtx, rows, cols, type, matrix_type)\
({\
    if (((mtx) = mtx_allocate(rows, cols, sizeof(type), matrix_type,\
            __FILE__, __LINE__, #mtx)) == NULL)\
        goto fail;\
})

#define mtx_double(mtx, rows, cols) mtx_init(mtx, rows, cols, double, vtype_double)
#define mtx_float(mtx, rows, cols) mtx_init(mtx, rows, cols, float, vtype_float)
#define mtx_int(mtx, rows, cols) mtx_init(mtx, rows, cols, int, vtype_int)

#define emtx(mtx, row, col, type)\
    *((type *) get_element(mtx, row, col))

#define dmtx(mtx, row, col)\
    *((double *) get_element(mtx, row, col))

#define fmtx(mtx, row, col)\
    *((float *) get_element(mtx, row, col))

#define imtx(mtx, row, col)\
    *((int *) get_element(mtx, row, col))

#endif
