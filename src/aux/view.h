#ifndef VIEW_H
#define VIEW_H

#include <stdio.h>

#include "common.h"
#include "array.h"

template<typename T>
class view {
    public:
        size_t ndim, *shape, *strides;

        view(arptr const arr)
        {
            size_t datasize = arr->datasize, ndim = arr->ndim;
            
            if (arr->isnumpy)
            {
                this->isnumpy = true;
                this->strides = new size_t[ndim];
                
                for(size_t ii = ndim; ii--;)
                {
                    this->strides[ii] = size_t(double(arr->strides[ii]) / datasize);
                }
            }
            else
                this->strides = arr->strides;
                
            this->ndim = arr->ndim;
            this->shape = arr->shape;
            data = static_cast<T*>(arr->data);
        }
        
        T* get_data() const {
            return this.data;
        }

        ~view()
        {
            if (this->isnumpy)
                delete[] this->strides;
        }
        
        T& operator()(size_t ii) {
            return this->data[ii * this->strides[0]];
        }

        T& operator()(size_t ii, size_t jj) {
            return this->data[ii * this->strides[0] + jj * this->strides[1]];
        }

        T& operator()(size_t ii, size_t jj, size_t kk) {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2]];
        }

        T& operator()(size_t ii, size_t jj, size_t kk, size_t ll) {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2] + ll * this->strides[3]];
        }

        T const& operator()(size_t ii) const {
            return this->data[ii * this->strides[0]];
        }

        T const& operator()(size_t ii, size_t jj) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1]];
        }

        T const& operator()(size_t ii, size_t jj, size_t kk) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2]];
        }

        T const& operator()(size_t ii, size_t jj, size_t kk, size_t ll) const {
            return this->data[ii * this->strides[0] + jj * this->strides[1] + 
                              kk * this->strides[2] + ll * this->strides[3]];
        }
    private:
        bool isnumpy;
        T* data;
};

extern_begin

typedef struct view_meta {
        int isnumpy;
        size_t ndim, *shape, *strides;
} view_meta;

#define m_def_view(TYPE, name) \
	typedef struct { view_meta md; TYPE* data; dtor dtor_; } name


#define __unpack(view) (void **) &((view).data), &((view).md)

void _setup_view(void **data, view_meta *md, arptr arr);


/* TODO error handling */
#define setup_view(view, arr)                           \
do {                                                    \
    if (((view) = m_malloc(sizeof(*(view)))) == NULL)   \
    {                                                   \
        goto fail;                                      \
    }                                                   \
    _setup_view(__unpack(view), (arr))                  \
} while(0)


m_def_view(double, view_double);
m_def_view(int, view_bool);


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

/*
void _setup_view(void **data, view_meta *md, nparray arr)
{
    md->ndim = arr->ndim;
    md->shape = arr->shape;
    md->strides = arr->strides;
    *data = PyArray_DATA(arr->npobj);
}

static void view_dtor(void* view)
{
    view_meta* view_ = (view_meta*) view;
    
    if (view_->isnumpy)
    {
        m_free(view_->strides);
        view_->strides = NULL;
    }
}

void _setup_view(void **data, view_meta *md, arptr arr)
{
    size_t ii = 0, datasize = arr->datasize;
    
    if (arr->isnumpy && (md->strides = m_new(size_t, arr->ndim)) != NULL)
    {
        m_forz(ii, ndim)
            md->strides[ii] = (size_t) ((double) arr->strides[ii] / datasize);
    }

    md->ndim = arr->ndim;
    md->shape = arr->shape;
    md->dtor_ = &view_dtor;
    *data = arr->data;
}
*/

#endif

extern_end

#endif
