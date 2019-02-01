#ifndef ARRAY_H
#define ARRAY_H

#include <string.h>

#include "utils.hpp"
#include "common.hpp"



struct ArrayMeta {
    enum class order { colmajor, rowmajor };
    
    enum class dtype { Unknown, Bool, Int, Long, Size_t,
                       Int8, Int16, Int32, Int64,
                       UInt8, UInt16, UInt32, UInt64,
                       Float32, Float64, Complex64, Complex128 };

    dtype type;
    order layout;
    size_t ndim, ndata, datasize, *shape, *strides;
    SMem mem;

    ArrayMeta() : type(dtype::Unknown), layout(order::rowmajor),
                  ndim(0), ndata(0), datasize(0), shape(nullptr),
                  strides(nullptr), mem(nullptr) {};
    
    ArrayMeta(std::initializer_list<size_t>& shape,
              dtype const type,
              order const layout);
    
    ArrayMeta(ArrayMeta const& meta);
    
    ArrayMeta operator=(ArrayMeta const& meta);

    ~ArrayMeta() = default;
};


class Array {
    using ptr = Array*;
    using AM = ArrayMeta;
     
    public:
        ArrayMeta meta;
        
        Array() = default;
        
        Array(std::initializer_list<size_t> const& shape,
              AM::dtype const type,
              AM::order const layout);
        
        int check_ndim(size_t const ndim) const;
        int check_type(int const type) const;
        int check_rows(size_t const rows) const;
        int check_cols(size_t const cols) const;
        
        void *get_data() const {
            return this->data;
        }
    private:
        void *data;
};

void setup_view(void** data, ArrayMeta& md, Array const& arr);

template<typename T>
class View {
    public:
        View(Array const& arr)
        {
            setup_view((void**) &(this->data), this->meta, arr);
        }
        
        T* get_data() const
        {
            return this->data;
        }

        ~View() = default;
        
        size_t const ndim() const
        {
            return this->meta.ndim;
        }
        
        size_t const shape(size_t ii) const
        {
            return this->meta.shape[ii];
        }

        T& operator()(size_t ii)
        {
            return this->data[calc_offset(this->meta.strides, ii)];
        }

        T& operator()(size_t ii,
                      size_t jj) {
            return this->data[calc_offset(this->meta.strides, ii, jj)];
        }

        T& operator()(size_t ii,
                      size_t jj,
                      size_t kk)
        {
            return this->data[calc_offset(this->meta.strides, ii, jj, kk)];
        }

        T& operator()(size_t ii,
                      size_t jj,
                      size_t kk,
                      size_t ll)
        {
            return this->data[calc_offset(this->meta.strides, ii, jj, kk, ll)];
        }

        
        T const& operator()(size_t ii) const
        {
            return this->data[calc_offset(this->meta.strides, ii)];
        }

        T const& operator()(size_t ii,
                            size_t jj) const
        {
            return this->data[calc_offset(this->meta.strides, ii, jj)];
        }

        T const& operator()(size_t ii,
                            size_t jj,
                            size_t kk) const
        {
            return this->data[calc_offset(this->meta.strides, ii, jj, kk)];
        }

        T const& operator()(size_t ii,
                            size_t jj,
                            size_t kk,
                            size_t ll) const
        {
            return this->data[calc_offset(this->meta.strides, ii, jj, kk, ll)];
        }
    private:
        ArrayMeta meta;
        T* data;
};


#ifdef m_get_impl

inline size_t const calc_offset(size_t const* strides,
                                size_t ii)
{
    return ii * strides[0];
}

inline size_t const calc_offset(size_t const* strides,
                                size_t ii,
                                size_t jj)
{
    return ii * strides[0] + jj * strides[1];
}

inline size_t const calc_offset(size_t const* strides,
                                size_t ii,
                                size_t jj,
                                size_t kk)
{
    return ii * strides[0] + jj * strides[1] + kk * strides[2];
}

inline size_t const calc_offset(size_t const* strides,
                                size_t ii,
                                size_t jj,
                                size_t kk,
                                size_t ll)
{
    return ii * strides[0] + jj * strides[1] + kk * strides[2] + ll * strides[3];
}


ArrayMeta(ArrayMeta const& meta)
{
    this->ndim = meta.ndim;
    this->ndata = meta.ndata;
    this->datasize = meta.datasize;
    this->shape = meta.shape;
    this->strides = meta.strides;
    this->mem = meta.mem;
}


ArrayMeta operator=(ArrayMeta const& meta) { return ArrayMeta{meta}; }


void setup_view(void**data,
                ArrayMeta& meta,
                Array const& arr)
{
    meta = arr.meta;
    *data = arr->get_data();
}

ArrayMeta::ArrayMeta()
{
    this->mem = make_smem(sizeof(size_t) * 2 * ndim + datasize * ndata); 
}


Array::Array(dtype const type, std::initializer_list<size_t> shape)
{
    this->meta{shape};    
}


int Array::check_ndim(size_t const ndim) const
{
    if (this->ndim != ndim)
    {
        // error
        return 1;
    }
    return 0;
}


int Array::check_type(int const type) const
{
    if (this->type != type)
    {
        // error
        return 1;
    }
    return 0;
}


int Array::check_rows(size_t const rows) const
{
    if (this->shape[0] != rows)
    {
        // error
        return 1;
    }
    return 0;
}


int Array::check_cols(size_t const cols) const
{
    if (this->shape[1] != cols)
    {
        // error
        return 1;
    }
    return 0;
}

static size_t const sizes[] = {
    [dtype::Unknown]       = 0,
    [dtype::Bool]          = sizeof(bool),
    [dtype::Int]           = sizeof(int),
    [dtype::Long]          = sizeof(long),
    [dtype::Size_t]        = sizeof(size_t),
    
    [dtype::Int8]          = sizeof(int8_t),
    [dtype::Int16]         = sizeof(int16_t),
    [dtype::Int32]         = sizeof(int32_t),
    [dtype::Int64]         = sizeof(int64_t),
    
    [dtype::UInt8]         = sizeof(uint8_t),
    [dtype::UInt16]        = sizeof(uint16_t),
    [dtype::UInt32]        = sizeof(uint32_t),
    [dtype::UInt64]        = sizeof(uint64_t),
    
    [dtype::Float32]       = sizeof(float),
    [dtype::Float64]       = sizeof(double),
    
    [dtype::Complex64]     = sizeof(complex<float>),
    [dtype::Complex128]    = sizeof(complex<double>)
};



/*

static void array_dtor(void *arr);

bool array_new(arptr* arr, dtype const type, size_t const ndim,
               layout const lay, size_t const* shape)
{
    arptr new = Mem_New(struct _array, 1);
    size_t ii = 0 total = 0;
    
    if (new == NULL) {
        Perror("array_new", "Memory allocation failed!\n");
        return true;
    }
    
    m_forz(ii, ndim)
        total += shape[ii];
    
    new->datasize = sizes[type];
    new->ndata = total;
    
    if ((new->shape = Mem_New(size_t, 2 * ndim + sizes[type] * total)) == NULL) {
        Perror("array_new", "Memory allocation failed!\n");
        Mem_Free(new);
        return true;
    }
    
    // check memcpy
    new->shape = memcpy(new->shape, shape, ndim * sizes[dt_size_t]);
    
    new->stride = new->shape + ndim;
    
    switch(lay) {
        case rowmajor:
            for(size_t ii = 0; ii < ndim; ++ii) {
                new->stride[ii] = 1;
                for(size_t jj = ii + 1; jj < ndim; ++jj) {
                    new->stride[ii] *= new->shape[jj];
                }
            }
            
            break;

        case colmajor:
            for(size_t ii = 0; ii < ndim; ++ii) {
                new->stride[ii] = 1;
                for(size_t jj = 1; jj < ii - 1; ++jj) {
                    new->stride[ii] *= new->shape[jj];
                }
            }
            
            break;
        //default:
            // error
    }
    
    new->data   = new->shape + 2 * ndim;
    new->dtor_  = array_dtor;
    
    *arr = new;
    
    return false;
}


bool array_read(arptr* arr, char const* path)
{
    arptr new = NULL;
    File infile = NULL;
    
    if (open(&infile, path, "rb"))
        goto fail;
    
    size_t doc_len = 0;
    char *doc;
    
    if (Readb(infile, sizes[dt_size_t], 1, &doc_len) < 0 or
        Readb(infile, sizes[dt_char], doc_len, doc) < 0)
        goto fail;
    
    size_t ndim = 0, ndata = 0, *shape = NULL;
    dtype type = 0;
    
    if (Readb(infile, sizes[dt_dtype], 1, &type) < 0 or
        Readb(infile, sizes[dt_size_t], 1, &ndim) < 0 or
        Readb(infile, sizes[dt_size_t], 1, &ndata) < 0 or
        Readb(infile, sizes[dt_size_t], ndim, shape) < 0 or)
        goto fail;
    
    if (array_new(&new, type, ndim, ,shape))
        goto fail;
    
    if (Readb(infile, sizes[dt_size_t], ndim, arr->stride) < 0 or
        Readb(infile, arr->datasize, ndata, arr->data) < 0)
        goto fail;
    
    *arr = new;
    del(infile);
    
    return false;

fail:
    del(arr); del(infile);
    return true;
}


bool array_write(arptr const arr, char const* path, char const* doc)
{
    File outfile = NULL;
    
    if (open(&outfile, path, "wb"))
        goto fail;
    
    size_t doc_len = strlen(doc);
    
    if(Writeb(outfile, sizes[s_size_t], 1, &doc_len) < 0 or
       Writeb(outfile, sizes[s_char], strlen(doc), doc) < 0)
       goto fail;
    
    size_t ndim = arr->ndim, ndata = arr->ndata;
    
    if (Writeb(outfile, sizes[dt_dtype], 1, &(arr->type)) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndim) < 0)
        goto fail;
        
    if (Writeb(outfile, sizes[dt_dtype], 1, &(arr->type)) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndim) < 0 or
        Writeb(outfile, sizes[dt_size_t], 1, &ndata) < 0 or
        Writeb(outfile, sizes[dt_size_t], ndim, arr->shape) < 0 or
        Writeb(outfile, sizes[dt_size_t], ndim, arr->stride) < 0 or
        Writeb(outfile, arr->datasize, ndata, arr->data) < 0)
        goto fail;
    
    del(outfile);
    return false;
    
fail:
    del(outfile);
    Perror("array_write", "Failed to write array to file: %s\n", path);
    return true;
}
*/
#endif

#endif
