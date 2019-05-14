#pragma once

#include <iostream>
#include <array>
#include <complex>
#include <memory>
#include <functional>

#include "type_info.hpp"


#define type_assert(T, P, msg) static_assert(T<P>::value, msg)
#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)


namespace aux {

using idx = long;

template<class T>
using ptr = T*;

template<class T>
using cptr = ptr<T> const;


// Dynamic number of dimensions
static idx constexpr Dynamic = -1;
static idx constexpr row = 0;
static idx constexpr col = 1;

static idx constexpr maxdim = 64;

static std::string const end = "\n";


// Forward declarations

struct ArrayInfo;

template<class T>
struct Array;


// typedef ptr<ArrayInfo> const inarray;
// typedef ptr<ArrayInfo> const outarray;

using inarray = ArrayInfo const&;
using outarray = ArrayInfo&;


using memtype = char;
using memptr = ptr<memtype>;


RTypeInfo const& type_info(int const type);
RTypeInfo const& type_info(dtype const type);


static void check_match(RTypeInfo const& one, RTypeInfo const& two)
{
    if (one.is_complex != two.is_complex) {
        throw std::runtime_error("Cannot convert complex to non complex "
                                 "type!");
    }
}


template<class T>
RTypeInfo const& type_info()
{
    return type_info(tpl2dtype<T>());
}


enum layout {
    colmajor,
    rowmajor
};


struct Memory 
{
    Memory(): _memory(nullptr), _size(0) {};
    Memory(idx const size);

    Memory(Memory const&) = delete;
    Memory(Memory&&) = delete;
    
    Memory& operator=(Memory const&) = default;
    Memory& operator=(Memory&&) = default;
    
    ~Memory() = default;
    
    void alloc(long size);
    
    memptr get() const noexcept;
    long size() const noexcept;

    std::unique_ptr<memtype[]> _memory;
    long _size;
};



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


template<class T>
struct View
{
    // make necessary items constant
    ptr<T> data;
    ptr<idx const> _shape;
    std::array<idx, maxdim> _strides;


    View() = default;
    
    View(View const&) = default;
    View(View&&) = default;
    
    View& operator=(View const&) = default;
    View& operator=(View&&) = default;
    
    ~View() = default;
    
    idx const& shape(idx const ii) const { return _shape[ii]; }
    
    T& operator()(idx const ii)
    {
        return data[ii * _strides[0]];
    }

    T& operator()(idx const ii, idx const jj)
    {
        return data[ii * _strides[0] + jj * _strides[1]];
    }

    T& operator()(idx const ii, idx const jj, idx const kk)
    {
        return data[ii * _strides[0] + jj * _strides[1] + kk * _strides[2]];
    }

    T& operator()(idx const ii, idx const jj, idx const kk, idx const ll)
    {
        return data[ii * _strides[0] + jj * _strides[1] + kk * _strides[2]
                    + ll * _strides[4]];
    }


    T const& operator()(idx const ii) const
    {
        return data[ii * _strides[0]];
    }

    T const& operator()(idx const ii, idx const jj) const
    {
        return data[ii * _strides[0] + jj * _strides[1]];
    }

    T const& operator()(idx const ii, idx const jj, idx const kk) const
    {
        return data[ii * _strides[0] + jj * _strides[1] + kk * _strides[2]];
    }

    T const& operator()(idx const ii, idx const jj, idx const kk, idx const ll) const
    {
        return data[ii * _strides[0] + jj * _strides[1] + kk * _strides[2]
                    + ll * _strides[4]];
    }
};


struct ArrayInfo {
    int const type, is_numpy;
    idx const ndim, ndata, datasize;
    ptr<idx const> const shape, strides;
    memptr data;
    
    ArrayInfo() = delete;
    ~ArrayInfo() = default;
    
    
    bool check_ndim(idx const ndim) const;
    bool check_shape(idx const nelem, idx dim = 0) const;


    
    RTypeInfo const& get_type() const
    {
        return type_info(type);
    }
    
    // Some sanity checks.
    
    template<class T>
    void basic_check(idx const ndim) const
    {
        if (ndim > maxdim) {
            throw std::runtime_error("Exceeded maximum number of dimensions!");
        }
        
        if (ndim < 0 and ndim != Dynamic) {
            throw std::runtime_error("ndim should be either a "
                                     "positive integer or Dynamic");
        }
        
        //type_assert(std::is_pod, T, "Type should be Plain Old Datatype!");
        
        type_assert(not std::is_void, T, "Type should not be void!");
        type_assert(not std::is_null_pointer, T, "Type should not be nullptr!");
        type_assert(not std::is_pointer, T, "Type should not be a pointer!");
        
        auto const _ndim = this->ndim;
        
        if (ndim != Dynamic and ndim != _ndim) {
            printf("View ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }


    template<class T>
    View<T> view(idx const ndim) const
    {
        basic_check<T>(ndim);
        
        auto const& arr_type = get_type(), req_type = type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("View id: %d, array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        
        check_match(arr_type, req_type);

        View<T> ret;
        ret._shape = shape;
        ret._strides = {0};
        
        if (is_numpy) {
            auto const _ndim = this->ndim;
            
            idx const dsize = datasize;
            
            for (idx ii = 0; ii < _ndim; ++ii) {
                ret._strides[ii] = idx(double(strides[ii]) / dsize);
            }
        }
        else {
            auto const _ndim = this->ndim;

            for (idx ii = 0; ii < _ndim; ++ii) {
                ret._strides[ii] = strides[ii];
            }
        }
        
        ret.data = reinterpret_cast<T*>(data);

        return ret;
    }


    template<class T>
    Array<T> const array(idx const ndim) const
    {
        return Array<T>(*this, ndim);
    }
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



#define make_convert(T, P) \
[](memptr in) { return static_cast<T>(*reinterpret_cast<P*>(in)); }

#define make_noconvert(T) \
[](memptr in) { return *reinterpret_cast<T*>(in); }


template<class T>
struct Array {
    using value_type = T;
    using convert_fun = std::function<value_type(memptr)>;
    
    
    ArrayInfo const& array;
    memptr const data;
    ptr<idx const> const strides;
    convert_fun const convert;

    Array() = delete;

    Array(Array const&) = default;
    Array(Array&&) = default;
    
    Array& operator=(Array const&) = default;
    Array& operator=(Array&&) = default;
    
    ~Array() = default;

    static convert_fun const converter_factory(int const type)
    {
        switch(static_cast<dtype>(type)) {
            case dtype::Int:
                return make_convert(T, int);
            case dtype::Long:
                return make_convert(T, long);
            case dtype::Size_t:
                return make_convert(T, size_t);
    
            case dtype::Int8:
                return make_convert(T, int8_t);
            case dtype::Int16:
                return make_convert(T, int16_t);
            case dtype::Int32:
                return make_convert(T, int32_t);
            case dtype::Int64:
                return make_convert(T, int64_t);

            case dtype::UInt8:
                return make_convert(T, uint8_t);
            case dtype::UInt16:
                return make_convert(T, uint16_t);
            case dtype::UInt32:
                return make_convert(T, uint32_t);
            case dtype::UInt64:
                return make_convert(T, uint64_t);
    
            case dtype::Float32:
                return make_convert(T, float);
            case dtype::Float64:
                return make_convert(T, double);

            //case dtype::Complex64:
                //return convert<T, cpx64>;
            //case dtype::Complex128:
                //return convert<T, cpx128>;        

            default:
                throw std::runtime_error("AA");
        }
    }

    
    explicit Array(inarray arr_ref, idx const ndim)
    :
    array(arr_ref), data(arr_ref.data), strides(arr_ref.strides),
    convert(converter_factory(array.type))
    {
        array.basic_check<T>(ndim);
        auto const& arr_type = array.get_type(), req_type = type_info<T>();
        check_match(arr_type, req_type);
    }

    
    T const operator ()(idx const ii)
    {
        return convert(data + ii * strides[0]);
    }
};


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
