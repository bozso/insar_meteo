#ifndef NUMPY_HPP
#define NUMPY_HPP

#include "Eigen/Core"
#include <cassert>

class Array;
using array_ptr = Array*;
using DStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template<class T>
using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using memtype = char;
using memptr = memtype*;

using cpx64  = std::complex<float>;
using cpx128 = std::complex<double>;


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


class Array {
public:
    typedef long idx;

    int type;
    idx ndim, ndata, datasize, *shape, *strides;
    memptr data;
};



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



#endif
