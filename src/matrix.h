#ifndef MATRIX_H
#define MATRIX_H

typedef enum data_type_t {
    data_double,
    data_float,
    data_int,
    data_other
} data_type;

typedef struct matrix_t {
    uint rows, cols;
    size_t elem_size;
    data_type type;
    char * data;
} matrix;


matrix * mtx_allocate(uint rows, uint cols, size_t elem_size, data_type type,
                      char * file, int line, char * matrix_name);

void mtx_safe_free(matrix * mtx);
void mtx_free(matrix * mtx);

inline char * get_element(matrix * mtx, uint row, uint col)
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

#endif
