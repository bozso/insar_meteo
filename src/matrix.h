#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdlib.h>
#include "aux_macros.h"

typedef unsigned int uint;

typedef struct matrix_t {
    uint rows, cols;
    size_t elem_size;
    void * data;
} matrix;

matrix * allocate_matrix(uint rows, uint cols, size_t elem_size);

#define init_matrix(mtx, rows, cols, type)\
({\
    (mtx) = allocate_matrix(rows, cols, sizeof(type));\
    if ((mtx)->data == NULL) {\
        errorln("FILE: %s LINE: %d. Allocation of matrix %s failed.", __FILE__,\
                 __LINE__, #mtx);\
        goto fail;\
    }\
})

#define melem(mtx, row, col, type)\
    *((type *) (mtx)->data + ((col) + (row) * (mtx)->cols) * (mtx)->elem_size)

#define free_matrix(mtx)\
({\
    if (mtx != NULL) {\
        free(mtx->data);\
        free(mtx);\
    }\
});

#endif
