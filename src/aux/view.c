#include "common.h"
#include "view.h"

void _setup_view(void **data, view_meta *md, array arr)
{
    md->ndim = arr->ndim;
    md->shape = arr->shape;
    md->stride = arr->stride;
    *data = arr->data;
}
