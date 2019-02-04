#include "array.hpp"

#include <string>
#include <fstream>

using std::string;
using std::ios::binary;

typedef Array::idx Aidx;


static Array Array::from_file(char const* data, char const* table)
{
    ifstream dataf{data, binary}, tablef{table};
    
    
    
    
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
    
    this->mem = make_shared(this->datasize * total 
                            + 2 * this->ndim * sizeof(Aidx));
    
    this->shape = static_cast<Aidx*>(this->mem.get())
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
