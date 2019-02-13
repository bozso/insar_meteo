#ifndef ARRAY_H
#define ARRAY_H

#include <string.h>

#include "utils.hpp"
#include "common.hpp"

template<typename T> struct View;

struct Array {
    typedef long idx;
    typedef Array* ptr;

    enum order { ColMajor, RowMajor };
    
    enum dtype { Unknown, Bool, Int, Long, Size_t,
                 Int8, Int16, Int32, Int64,
                 UInt8, UInt16, UInt32, UInt64,
                 Float32, Float64, Complex64, Complex128 };

    dtype type;
    order layout;
    idx ndim, ndata, datasize, *shape, *strides;
    SharedMemory::memory_type* data;
    SharedMemory memory;

    Array() : type(Unknown), layout(RowMajor), ndim(0), ndata(0), datasize(0),
              shape(nullptr), strides(nullptr), memory() {};
    
    Array(dtype const type,
          std::initializer_list<idx>& shape,
          order const layout = RowMajor);
    
    static Array load(char const* data,
                      char const* table);

    void save(char const* data,
              char const* table);
    
    Array(Array const& arr);
    Array operator=(Array const& arr);

    template<class T>
    View<T> view() const
    {
        View<T> ret{*this};
        return ret;
    }
    
    
    ~Array() = default;


    static idx const calc_offset(idx const* strides,
                                 idx ii);
    
    static idx const calc_offset(idx const* strides,
                                 idx ii,
                                 idx jj);
    
    static idx const calc_offset(idx const* strides,
                                 idx ii,
                                 idx jj,
                                 idx kk);
    
    static idx const calc_offset(idx const* strides,
                                        idx ii,
                                        idx jj,
                                        idx kk,
                                        idx ll);
};


template<typename T>
struct View {
    View(Array const& arr)
    {
        this->arr = arr;
        data = (T*) arr.data;
    }
    
    T* get_data() const { return this->data; }

    ~View() = default;
    
    Array::idx ndim() const
    {
        return this->arr.ndim;
    }
    
    Array::idx shape(Array::idx ii) const
    {
        return this->arr.shape[ii];
    }

    T& operator()(Array::idx ii)
    {
        return this->data[Array::calc_offset(this->arr.strides, ii)];
    }

    T& operator()(Array::idx ii,
                  Array::idx jj)
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj)];
    }

    T& operator()(Array::idx ii,
                  Array::idx jj,
                  Array::idx kk)
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj, kk)];
    }

    T& operator()(Array::idx ii,
                  Array::idx jj,
                  Array::idx kk,
                  Array::idx ll)
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj, kk, ll)];
    }

    
    T const& operator()(Array::idx ii) const
    {
        return this->data[Array::calc_offset(this->arr.strides, ii)];
    }

    T const& operator()(Array::idx ii,
                        Array::idx jj) const
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj)];
    }

    T const& operator()(Array::idx ii,
                        Array::idx jj,
                        Array::idx kk) const
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj, kk)];
    }

    T const& operator()(Array::idx ii,
                        Array::idx jj,
                        Array::idx kk,
                        Array::idx ll) const
    {
        return this->data[Array::calc_offset(this->arr.strides, ii, jj, kk, ll)];
    }
    private:
        Array& arr;
        T* data;
};


#ifdef m_get_impl

#include <string>
#include <fstream>

using std::string;
using std::ios::binary;

typedef Array::idx Aidx;


static Array Array::load(char const* data, char const* table)
{
    ifstream dataf{data, binary}, tablef{table};
    
    Array ret;
    
    return ret;
}

void Array::save(char const* data, char const* table)
{
    
    
}


Array::Array(Array::dtype const type,
             std::initializer_list<Aidx>& shape,
             Array::order const layout = Array::RowMajor)
{
    idx total = 0;
    
    for (auto const& sh: shape)
    {
        total += sh;
    }
    
    this->datasize = sizes[type];
    this->ndim = shape.size();
    this->ndata = total;
    
    this->memory(this->datasize * total + 2 * this->ndim * sizeof(Aidx))
    
    this->shape = static_cast<Aidx*>(this->memory.ptr() )
    this->stride = this->shape + ndim;
    this->data = static_cast<mem_var*>(this->strides + ndim);
    
    
    switch(layout) {
        case Array::RowMajor:
            for(Aidx ii = 0; ii < ndim; ++ii)
            {
                this->stride[ii] = 1;
                for(Aidx jj = ii + 1; jj < ndim; ++jj)
                {
                    this->stride[ii] *= this->shape[jj];
                }
            }
            break;

        case Array::ColMajor:
            for(Aidx ii = 0; ii < ndim; ++ii)
            {
                this->stride[ii] = 1;
                for(Aidx jj = 1; jj < ii - 1; ++jj)
                {
                    this->stride[ii] *= this->shape[jj];
                }
            }
            
            break;
        //default:
            // error
    }
}


Array::Array(Array const& arr)
{
    this->ndim = arr.ndim;
    this->ndata = arr.ndata;
    this->datasize = arr.datasize;
    this->shape = arr.shape;
    this->strides = arr.strides;
    this->data = arr.data;
    this->mem = arr.mem;
}


Array Array::operator=(Array const& arr) { return Array{arr}; }


bool Array::check_ndim(Aidx const ndim) const
{
    if (this->ndim != ndim)
    {
        // error
        return true;
    }
    return false;
}


bool Array::check_type(Array::dtype const type) const
{
    if (this->type != type)
    {
        // error
        return true;
    }
    return false;
}


bool Array::check_rows(Aidx const rows) const
{
    if (this->shape[0] != rows)
    {
        // error
        return true;
    }
    return false;
}


bool Array::check_cols(Aidx const cols) const
{
    if (this->shape[1] != cols)
    {
        // error
        return true;
    }
    return false;
}


static Array::idx const sizes[] = {
    [Array::Unknown]       = 0,
    [Array::Bool]          = sizeof(bool),
    [Array::Int]           = sizeof(int),
    [Array::Long]          = sizeof(long),
    [Array::Size_t]        = sizeof(size_t),
    
    [Array::Int8]          = sizeof(int8_t),
    [Array::Int16]         = sizeof(int16_t),
    [Array::Int32]         = sizeof(int32_t),
    [Array::Int64]         = sizeof(int64_t),
    
    [Array::UInt8]         = sizeof(uint8_t),
    [Array::UInt16]        = sizeof(uint16_t),
    [Array::UInt32]        = sizeof(uint32_t),
    [Array::UInt64]        = sizeof(uint64_t),
    
    [Array::Float32]       = sizeof(float),
    [Array::Float64]       = sizeof(double),
    
    [Array::Complex64]     = sizeof(complex<float>),
    [Array::Complex128]    = sizeof(complex<double>)
};


static inline Aidx const Array::calc_offset(Aidx const* strides,
                                            Aidx ii)
{
    return ii * strides[0];
}

static inline Aidx const Array::calc_offset(Aidx const* strides,
                                            Aidx ii,
                                            Aidx jj)
{
    return ii * strides[0] + jj * strides[1];
}

static inline Aidx const Array::calc_offset(Aidx const* strides,
                                            Aidx ii,
                                            Aidx jj,
                                            Aidx kk)
{
    return ii * strides[0] + jj * strides[1] + kk * strides[2];
}

static inline Aidx const Array::calc_offset(Aidx const* strides,
                                            Aidx ii,
                                            Aidx jj,
                                            Aidx kk,
                                            Aidx ll)
{
    return ii * strides[0] + jj * strides[1] + kk * strides[2] + ll * strides[3];
}

#endif

/*

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
