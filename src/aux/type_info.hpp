#ifndef AUX_TYPEINFO_HPP
#define AUX_TYPEINFO_HPP

#include <type_traits>
#include <complex>


namespace std {

template<class T>
struct is_complex : std::false_type {};


template<> struct is_complex<complex<float>> : std::true_type {};
template<> struct is_complex<complex<double>> : std::true_type {};

}


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


typedef std::complex<float>  cpxf;
typedef std::complex<double> cpxd;


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


struct RTypeInfo {
    typedef char const*const name_t;
    
    bool const is_pointer, is_void, is_complex, is_float, is_scalar,
               is_arithmetic, is_pod;
    size_t const size;
    int const id;
    name_t name;
    
    RTypeInfo() : is_pointer(false), is_void(false), is_complex(false),
                  is_float(false), is_scalar(false), is_arithmetic(false),
                  is_pod(false), size(0), id(0), name(nullptr) {}
    ~RTypeInfo() = default;
    
    
    RTypeInfo(bool is_pointer, bool is_void, bool is_complex, bool is_float,
              bool is_scalar, bool is_arithmetic, bool is_pod, size_t size,
              int id, name_t name = "Unknown") :
                is_pointer(is_pointer), is_void(is_void),
                is_complex(is_complex), is_float(is_float),
                is_scalar(is_scalar), is_arithmetic(is_arithmetic),
                is_pod(is_pod), size(size), id(id), name(name) {}
    
    
    template<class T>
    static RTypeInfo make_info(name_t name)
    {
        return RTypeInfo(std::is_pointer<T>::value, std::is_void<T>::value,
                         std::is_complex<T>::value,
                         std::is_floating_point<T>::value,
                         std::is_scalar<T>::value,
                         std::is_arithmetic<T>::value,
                         std::is_pod<T>::value,
                         sizeof(T), tpl2dtype<T>(), name);
    }
    
    bool operator==(RTypeInfo const& other) const { return id == other.id; }
};


// aux namespace
}

// guard
#endif
