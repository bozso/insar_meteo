#ifndef __TYPE_INFO_HPP
#define __TYPE_INFO_HPP

#include <type_traits>
#include <complex>

namespace aux {



enum class dtype : int {
    Unknown     = 0,
    Int         = 1,
    Long        = 2,
    Size_t      = 3,
    Int8        = 4,
    Int16       = 5,
    Int32       = 6,
    Int64       = 7,
    UInt8       = 8,
    UInt16      = 9,
    UInt32      = 10,
    UInt64      = 11,
    Float32     = 12,
    Float64     = 13,
    Complex64   = 14,
    Complex128  = 15
};




template<class T>
int const tpl2dtype()
{
    if (std::is_same<T, int>::value)
        return static_cast<int>(dtype::Int);
    else if (std::is_same<T, long>::value)
        return static_cast<int>(dtype::Long);
    else if (std::is_same<T, size_t>::value)
        return static_cast<int>(dtype::Size_t);

    else if (std::is_same<T, int8_t>::value)
        return static_cast<int>(dtype::Int8);
    else if (std::is_same<T, int16_t>::value)
        return static_cast<int>(dtype::Int16);
    else if (std::is_same<T, int32_t>::value)
        return static_cast<int>(dtype::Int32);
    else if (std::is_same<T, int64_t>::value)
        return static_cast<int>(dtype::Int64);

    else if (std::is_same<T, uint8_t>::value)
        return static_cast<int>(dtype::UInt8);
    else if (std::is_same<T, uint16_t>::value)
        return static_cast<int>(dtype::UInt16);
    else if (std::is_same<T, uint32_t>::value)
        return static_cast<int>(dtype::UInt32);
    else if (std::is_same<T, uint64_t>::value)
        return static_cast<int>(dtype::UInt64);

    else if (std::is_same<T, float>::value)
        return static_cast<int>(dtype::Float32);
    else if (std::is_same<T, double>::value)
        return static_cast<int>(dtype::Float64);

    else if (std::is_same<T, cpxf>::value)
        return static_cast<int>(dtype::Complex64);
    else if (std::is_same<T, cpxd>::value)
        return static_cast<int>(dtype::Complex128);
}




// aux namespace
}

#endif