#ifndef VIEW_H
#define VIEW_H

#include "array.h"
#include "common.h"

extern_begin

typedef struct view_meta {
        size_t ndim, *shape, *strides;
} view_meta;

#define m_def_view(TYPE, name) \
	typedef struct { TYPE* data; view_meta md; } name;

#define __unpack(view) (void **) &((view).data), &((view).md)

void _setup_view(void **data, view_meta *md, arrayptr arr);

#define setup_view(view, arr) _setup_view(__unpack(view), (arr))


m_def_view(double, view_double)
m_def_view(dt_bool, view_bool)


#define ar_elem1(ar_struct, ii)\
        (ar_struct).data[(ii) * (ar_struct).md.strides[0]]

#define ar_elem2(ar_struct, ii, jj)\
        (ar_struct).data[  (ii) * (ar_struct).md.strides[0]\
                         + (jj) * (ar_struct).md.strides[1]]

#define ar_elem3(ar_struct, ii, jj, kk)\
        (ar_struct).data[  (ii) * (ar_struct).md.strides[0]\
                         + (jj) * (ar_struct).md.strides[1]\
                         + (kk) * (ar_struct).md.strides[2]]


#define ar_ptr1(ar_struct, ii)\
        (ar_struct).data + (ii) * (ar_struct).md.strides[0]


#define ar_ptr2(ar_struct, ii, jj)\
        (ar_struct).data + (ii) * (ar_struct).md.strides[0]\
                         + (jj) * (ar_struct).md.strides[1]


#define ar_ptr3(ar_struct, ii, jj, kk)\
        (ar_struct).data + (ii) * (ar_struct).md.strides[0]\
                         + (jj) * (ar_struct).md.strides[1]\
                         + (kk) * (ar_struct).md.strides[2]


#ifdef m_get_impl

void _setup_view(void **data, view_meta *md, arrayptr arr)
{
    md->ndim = arr->ndim;
    md->shape = arr->shape;
    md->strides = arr->strides;
    *data = arr->data;
}

#endif

extern_end

#endif
