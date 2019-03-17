#ifndef NUMPY_HPP
#define NUMPY_HPP

#include <complex>
#include <memory>

namespace numpy
{

struct Array;
typedef Array* array_ptr;
typedef long idx;

typedef char memtype;
typedef memtype* memptr;

typedef std::complex<float> cpx64;
typedef std::complex<double> cpx128;


enum class dtype : int {
    Unknown     = 0,
    Bool        = 1,
    Int         = 2,
    Long        = 3,
    Size_t      = 4,
    Int8        = 5,
    Int16       = 6,
    Int32       = 7,
    Int64       = 8,
    UInt8       = 9,
    UInt16      = 10,
    UInt32      = 11,
    UInt64      = 12,
    Float32     = 13,
    Float64     = 14,
    Complex64   = 15,
    Complex128  = 16
};


enum layout {
    colmajor,
    rowmajor
};



struct Array
{
    int type, is_numpy;
    idx ndim, ndata, datasize, *shape, *strides;
    memptr data;
};


static std::unique_ptr<memtype[]> make_memory(long size)
{
    return std::unique_ptr<memtype[]>(new memtype[size]);
}


struct Data
{
    Data() = delete;
    Data(memptr ptr) : _data(ptr), owned(false) {};
    Data(long size) :  _data(size), owned(true) {};
    
    ~Data() { dtor(); }
    
    memptr get_data() const
    {
        if (owned)
        {
            return _data.uptr.get();
        }
        else
        {
            return _data.ptr;
        }
    }
    
private:
    union data
    {
        memptr ptr;
        std::unique_ptr<memtype[]> uptr;        

        data() : ptr(nullptr) {};
        data(memptr ptr) : ptr(ptr) {};
        data(long size) : uptr(make_memory(size)) {};
        
        ~data() {};
        
    } _data;
    bool owned;
    
    void dtor()
    {
        if (owned)
        {
            _data.uptr.~unique_ptr<memtype[]>();
        }
    }
};


template<class T>
struct View
{
private:
    Data _data_;
    T* _data;
    idx ndim, *shape;
    std::unique_ptr<idx[]> strides;
};



/*
using DStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template<class T>
using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;



template<class T1, class T2>
DMatrix<T1> convert(array_ptr arr)
{
    auto dsize = arr->datasize;
    auto stride = DStride(arr->strides[0] / dsize, arr->strides[1] / dsize);
    
    auto data = reinterpret_cast<T2*>(arr->data);
    auto tmp = \
    Eigen::Map<DMatrix<T2>, Eigen::Unaligned, DStride>(data, arr->shape[0],
                                                             arr->shape[1],
                                                             stride);
    
    return tmp.template cast <T1>();
}



template<class T>
DMatrix<T> from_numpy(array_ptr arr, bool colvec = true)
{
    assert(arr->ndim == 2);
    
    switch(arr->type)
    {
        case 1:
            return convert<T, bool>(arr);
        case 2:
            return convert<T, int>(arr);
        case 3:
            return convert<T, long>(arr);
        case 4:
            return convert<T, size_t>(arr);

        case 5:
            return convert<T, int8_t>(arr);
        case 6:
            return convert<T, int16_t>(arr);
        case 7:
            return convert<T, int32_t>(arr);
        case 8:
            return convert<T, int64_t>(arr);

        case 9:
            return convert<T, uint8_t>(arr);
        case 10:
            return convert<T, uint16_t>(arr);
        case 11:
            return convert<T, uint32_t>(arr);
        case 12:
            return convert<T, uint64_t>(arr);

        case 13:
            return convert<T, float>(arr);
        case 14:
            return convert<T, double>(arr);

        //case 15:
            //return convert<T, cpx64>(arr);
        //case 16:
            //return convert<T, cpx128>(arr);
    }
}
*/

// namespace
}

#endif
