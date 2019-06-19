#ifndef __AUX_HPP
#define __AUX_HPP

#include "type_info.hpp"


#define type_assert(T, P, msg) static_assert(T<P>::value, msg)
#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)


namespace aux {

template<class T>
using ptr = T*;

template<class T>
using cptr = ptr<T> const;

static std::string const end = "\n";

using memtype = char;
using memptr = ptr<memtype>;


RTypeInfo const& type_info(int const type);
RTypeInfo const& type_info(dtype const type);


template<class T>
RTypeInfo const& type_info()
{
    return type_info(tpl2dtype<T>());
}


template<class T>
struct Ptr
{
    using ptr_t = T*;
    using value_type = T;
    
    ptr_t _ptr{nullptr};
    
    
    Ptr() = default;
    ~Ptr() = default;
    
    Ptr(Ptr const&) = default;
    Ptr& operator=(Ptr const&) = default;
    
    Ptr(ptr_t const ptr) : _ptr{ptr} {}
    Ptr& operator=(ptr_t const ptr) { _ptr = ptr; return *this; }
    
    Ptr(Ptr&&) = delete;
    Ptr& operator=(Ptr&&) = delete;
    
    Ptr(ptr_t&&) = delete;
    Ptr& operator=(ptr_t&&) = delete;
    
    ptr_t get() { return _ptr; }
    
    value_type& operator*() { return *_ptr; }
    value_type const& operator*() const { return *_ptr; }
    
    ptr_t operator->() { return _ptr; }
    ptr_t const operator->() const { return _ptr; }
};

// TODO: better converting with std::function

/*
template<class T, class P>
struct converter {
    T operator ()(memptr in) {
        return static_cast<T>(*reinterpret_cast<P*>(in));
    }
};


template<class T, class P>
static T convert(memptr in)
{
    return static_cast<T>(*reinterpret_cast<P*>(in));
};
*/


#define switcher(f, type_id, ...)                      \
do {                                                   \
    switch (static_cast<dtype>((type_id))) {           \
        case dtype::Int:                               \
            f<int>(__VA_ARGS__);                       \
            break;                                     \
        case dtype::Long:                              \
            f<long>(__VA_ARGS__);                      \
            break;                                     \
        case dtype::Size_t:                            \
            f<size_t>(__VA_ARGS__);                    \
            break;                                     \
                                                       \
        case dtype::Int8:                              \
            f<int8_t>(__VA_ARGS__);                    \
            break;                                     \
        case dtype::Int16:                             \
            f<int16_t>(__VA_ARGS__);                   \
            break;                                     \
        case dtype::Int32:                             \
            f<int32_t>(__VA_ARGS__);                   \
            break;                                     \
        case dtype::Int64:                             \
            f<int64_t>(__VA_ARGS__);                   \
            break;                                     \
                                                       \
                                                       \
        case dtype::UInt8:                             \
            f<uint8_t>(__VA_ARGS__);                   \
            break;                                     \
        case dtype::UInt16:                            \
            f<uint16_t>(__VA_ARGS__);                  \
            break;                                     \
        case dtype::UInt32:                            \
            f<uint32_t>(__VA_ARGS__);                  \
            break;                                     \
        case dtype::UInt64:                            \
            f<uint64_t>(__VA_ARGS__);                  \
            break;                                     \
                                                       \
        case dtype::Float32:                           \
            f<float>(__VA_ARGS__);                     \
            break;                                     \
        case dtype::Float64:                           \
            f<double>(__VA_ARGS__);                    \
            break;                                     \
                                                       \
        case dtype::Complex64:                         \
            f<cpxf>(__VA_ARGS__);                      \
            break;                                     \
        case dtype::Complex128:                        \
            f<cpxd>(__VA_ARGS__);                      \
            break;                                     \
                                                       \
        case dtype::Unknown:                           \
            throw std::runtime_error("Unknown type!"); \
            break;                                     \
    }                                                  \
} while(0)


/*
template<class Fun, class... Args>
void switcher(int const type_id, Fun f, Args&&... args)
{
    switch(static_cast<dtype>(type_id)) {
        case dtype::Int:
            f<int>(std::forward<Args>(args)...);
            break;
        case dtype::Long:
            f<long>(std::forward<Args>(args)...);
            break;
        case dtype::Size_t:
            f<size_t>(std::forward<Args>(args)...);
            break;
        
        case dtype::Int8:
            f<int8_t>(std::forward<Args>(args)...);
            break;

        case dtype::Int16:
            f<int16_t>(std::forward<Args>(args)...);
            break;

        case dtype::Int32:
            f<int32_t>(std::forward<Args>(args)...);
            break;

        case dtype::Int64:
            f<int64_t>(std::forward<Args>(args)...);
            break;
            

        case dtype::UInt8:
            f<uint8_t>(std::forward<Args>(args)...);
            break;

        case dtype::UInt16:
            f<uint16_t>(std::forward<Args>(args)...);
            break;

        case dtype::UInt32:
            f<uint32_t>(std::forward<Args>(args)...);
            break;

        case dtype::UInt64:
            f<uint64_t>(std::forward<Args>(args)...);
            break;
   
        case dtype::Float32:
            f<float>(std::forward<Args>(args)...);
            break;
   
        case dtype::Float64:
            f<double>(std::forward<Args>(args)...);
            break;

        case dtype::Complex64:
            f<cpxf>(std::forward<Args>(args)...);
            break;

        case dtype::Complex128:
            f<cpxd>(std::forward<Args>(args)...);
            break;

        case dtype::Unknown:
            throw std::runtime_error("Unknown type!");
            break;
    }
}
*/


// aux namespace
}


#endif