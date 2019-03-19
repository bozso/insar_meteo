#ifndef AUX_TYPEINFO_HPP
#define AUX_TYPEINFO_HPP

#include <type_traits>

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


typedef std::complex<float>  cpx64;
typedef std::complex<double> cpx128;


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

    else if (std::is_same<T, cpx64>::value)
        return static_cast<int>(dtype::Complex64);
    else if (std::is_same<T, cpx128>::value)
        return static_cast<int>(dtype::Complex128);
}


struct RTypeInfo {
    bool const is_pointer, is_void, is_complex, is_float, is_scalar,
               is_arithmetic, is_pod;
    size_t const size;
    int const id;
    
    RTypeInfo() : is_pointer(false), is_void(false), is_complex(false),
                  is_float(false), is_scalar(false), is_arithmetic(false),
                  is_pod(false), size(0), id(0) {}

    RTypeInfo(bool is_pointer, bool is_void, bool is_complex, bool is_float,
              bool is_scalar, bool is_arithmetic, bool is_pod, size_t size,
              int id) :
                is_pointer(is_pointer), is_void(is_void),
                is_complex(is_complex), is_float(is_float),
                is_scalar(is_scalar), is_arithmetic(is_arithmetic),
                is_pod(is_pod), size(size), id(id) {}
            
    ~RTypeInfo() = default;
};


template<class T>
struct _is_complex : std::false_type {};

template<> struct _is_complex<cpx64> : std::true_type {};
template<> struct _is_complex<cpx128> : std::true_type {};


template<class T>
struct TypeInfo {
    static constexpr bool is_pointer = std::is_pointer<T>::value;
    static constexpr bool is_void = std::is_void<T>::value;
    static bool constexpr is_complex = _is_complex<T>::value;
    static bool constexpr is_float = std::is_floating_point<T>::value;
    static bool constexpr is_scalar = std::is_scalar<T>::value;
    static bool constexpr is_arithmetic = std::is_arithmetic<T>::value;
    static bool constexpr is_pod = std::is_pod<T>::value;
    static size_t constexpr size = sizeof(T);
    
    static RTypeInfo make_info()
    {
        return RTypeInfo(is_pointer, is_void, is_complex, is_float, is_scalar,
                         is_arithmetic, is_pod, size, tpl2dtype<T>());
    }    
};


// aux namespace
}

// guard
#endif
