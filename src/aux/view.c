#include "common.h"
#include "view.h"

void _setup_view(void **data, view_meta *md, nparray *arr)
{
    md->ndim = arr->ndim;
    md->shape = arr->shape;
    md->strides = arr->strides;
    
    *data = PyArray_DATA(arr->npobj);
}
