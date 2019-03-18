#ifndef AUX_TYPEINFO_HPP
#define AUX_TYPEINFO_HPP

#include <type_traits>

namespace aux {

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


typedef std::complex<float>  cpx64;
typedef std::complex<double> cpx128;


struct _typeinfo {
    //static constexpr bool is_pointer = std::is_pointer<T>::value;
    //static constexpr bool is_void = std::is_void<T>::value;
    //static constexpr bool is_complex = false;
    //static constexpr bool is_float = std::is_floating_point<T>::value;
    //static constexpr bool is_scalar = std::is_scalar<T>::value;
    //static constexpr bool is_arithmetic = std::is_arithmetic<T>::value;
    //static constexpr bool is_pod = std::is_pod<T>::value;
    
    //static constexpr size_t size = sizeof(T);
};


struct TypeInfo {
    
    int id;
    
    TypeInfo(int const type_id) : id(type_id) {};
    
    template<class T>    
    static TypeInfo make_info()
    {
        if (std::is_same<T, bool>::value)
            return TypeInfo(static_cast<int>(dtype::Bool));
        else if (std::is_same<T, int>::value)
            return TypeInfo(static_cast<int>(dtype::Int));
        else
            return TypeInfo(0);
        /*
        else if (std::is_same<T, int>::value)
            return { static_cast<int>(dtype::Int) };
        else if (std::is_same<T, long>::value)
            id = static_cast<int>(dtype::Long);
        else if (std::is_same<T, size_t>::value)
            id = static_cast<int>(dtype::Size_t);
    
        else if (std::is_same<T, int8_t>::value)
            id = static_cast<int>(dtype::Int8);
        else if (std::is_same<T, int16_t>::value)
            id = static_cast<int>(dtype::Int16);
        else if (std::is_same<T, int32_t>::value)
            id = static_cast<int>(dtype::Int32);
        else if (std::is_same<T, int64_t>::value)
            id = static_cast<int>(dtype::Int64);
    
        else if (std::is_same<T, uint8_t>::value)
            id = static_cast<int>(dtype::UInt8);
        else if (std::is_same<T, uint16_t>::value)
            id = static_cast<int>(dtype::UInt16);
        else if (std::is_same<T, uint32_t>::value)
            id = static_cast<int>(dtype::UInt32);
        else if (std::is_same<T, uint64_t>::value)
            id = static_cast<int>(dtype::UInt64);
    
        else if (std::is_same<T, float>::value)
            id = static_cast<int>(dtype::Float32);
        else if (std::is_same<T, double>::value)
            id = static_cast<int>(dtype::Float64);
    
        else if (std::is_same<T, cpx64>::value)
            id = static_cast<int>(dtype::Complex64);
        else if (std::is_same<T, cpx128>::value)
            id = static_cast<int>(dtype::Complex128);
        */
        
        /*
        if (std::is_same<T, bool>::value)
            id = static_cast<int>(dtype::Bool);
        else if (std::is_same<T, int>::value)
            id = static_cast<int>(dtype::Int);
        else if (std::is_same<T, long>::value)
            id = static_cast<int>(dtype::Long);
        else if (std::is_same<T, size_t>::value)
            id = static_cast<int>(dtype::Size_t);
    
        else if (std::is_same<T, int8_t>::value)
            id = static_cast<int>(dtype::Int8);
        else if (std::is_same<T, int16_t>::value)
            id = static_cast<int>(dtype::Int16);
        else if (std::is_same<T, int32_t>::value)
            id = static_cast<int>(dtype::Int32);
        else if (std::is_same<T, int64_t>::value)
            id = static_cast<int>(dtype::Int64);
    
        else if (std::is_same<T, uint8_t>::value)
            id = static_cast<int>(dtype::UInt8);
        else if (std::is_same<T, uint16_t>::value)
            id = static_cast<int>(dtype::UInt16);
        else if (std::is_same<T, uint32_t>::value)
            id = static_cast<int>(dtype::UInt32);
        else if (std::is_same<T, uint64_t>::value)
            id = static_cast<int>(dtype::UInt64);
    
        else if (std::is_same<T, float>::value)
            id = static_cast<int>(dtype::Float32);
        else if (std::is_same<T, double>::value)
            id = static_cast<int>(dtype::Float64);
    
        else if (std::is_same<T, cpx64>::value)
            id = static_cast<int>(dtype::Complex64);
        else if (std::is_same<T, cpx128>::value)
            id = static_cast<int>(dtype::Complex128);
        */
    }
    ~TypeInfo() = default;
};


/*
template<> struct TypeInfo<bool>     { static constexpr int id = 2; };
template<> struct TypeInfo<int>      { static constexpr int id = 2; };
template<> struct TypeInfo<long>     { static constexpr int id = 3; };
template<> struct TypeInfo<size_t>   { static constexpr int id = 4; };

template<> struct TypeInfo<int8_t>   { static constexpr int id = 5; };
template<> struct TypeInfo<int16_t>  { static constexpr int id = 6; };
template<> struct TypeInfo<int32_t>  { static constexpr int id = 7; };
template<> struct TypeInfo<int64_t>  { static constexpr int id = 8; };

template<> struct TypeInfo<uint8_t>  { static constexpr int id = 9; };
template<> struct TypeInfo<uint16_t> { static constexpr int id = 10; };
template<> struct TypeInfo<uint32_t> { static constexpr int id = 11; };
template<> struct TypeInfo<uint64_t> { static constexpr int id = 12; };

template<> struct TypeInfo<float>    { static constexpr int id = 13; };
template<> struct TypeInfo<double>   { static constexpr int id = 14; };

template<> struct TypeInfo<cpx64>    { static constexpr int id = 15; };
template<> struct TypeInfo<cpx128>   { static constexpr int id = 16; };

*/

//template<> struct TypeInfo<cpx64>    { static constexpr bool is_complex = true; };
//template<> struct TypeInfo<cpx128>   { static constexpr bool is_complex = true; };

// aux namespace
}

// guard
#endif

/*

Long      
Size_t    
Int8       
Int16      
Int32     }
Int64     
UInt8     /
UInt16    }
UInt32    
UInt64    /
Float32   #
Float64   
Complex64 
Complex128

*/
