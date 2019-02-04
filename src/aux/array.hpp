#ifndef ARRAY_H
#define ARRAY_H

#include <string.h>

#include "utils.hpp"
#include "common.hpp"

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
    mem_var *data;
    shared_mem mem;

    Array() : type(Unknown), layout(RowMajor), ndim(0), ndata(0), datasize(0),
              shape(nullptr), strides(nullptr), mem(nullptr) {};
    
    Array::Array(dtype const type,
                 std::initializer_list<idx>& shape,
                 order const layout = RowMajor);
    
    Array(Array const& arr);
    Array operator=(Array const& arr);

    template<class T>
    View<T> view() const
    {
        View<T> ret{*this};
        return ret;
    }
    
    
    ~Array() = default;


    static inline idx const calc_offset(idx const* strides,
                                        idx ii);
    
    static inline idx const calc_offset(idx const* strides,
                                        idx ii,
                                        idx jj);
    
    static inline idx const calc_offset(idx const* strides,
                                        idx ii,
                                        idx jj,
                                        idx kk);
    
    static inline idx const calc_offset(idx const* strides,
                                        idx ii,
                                        idx jj,
                                        idx kk,
                                        idx ll);
};


template<typename T>
struct View {
    View(Array const& arr) { this->arr = arr; data = arr.get_data() }
    
    T* get_data() const { return this->data; }

    ~View() = default;
    
    Array::idx const ndim() const
    {
        return this->arr.ndim;
    }
    
    Array::idx const shape(Array::idx ii) const
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
