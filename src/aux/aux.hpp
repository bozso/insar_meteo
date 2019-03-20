#ifndef AUX_HPP
#define AUX_HPP

#include <iostream>
#include <complex>
#include <memory>
#include <functional>

#include "type_info.hpp"

namespace aux {

// from https://www.modernescpp.com/index.php/c-core-guidelines-rules-for-variadic-templates

/*
void print(const char* format)
{
    std::cout << format;
}

 
template<typename T, typename ... Args>
void print(const char* format, T value, Args ... args)
{
    for ( ; *format != '\0'; format++ ) {
        if ( *format == '%' ) {
           std::cout << value;
           print(format + 1, args ... );
           return;
        }
        std::cout << *format;
    }
}
*/


template<class T>
using ptr = T*;

typedef long idx;

// Dynamic number of dimensions
static idx constexpr Dynamic = -1;


// Forward declarations

struct ArrayInfo;

typedef ptr<ArrayInfo> array_ptr;

typedef char memtype;
typedef ptr<memtype> memptr;


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



template<class T1, class T2>
static T1 convert(memptr in)
{
    return static_cast<T1>(*reinterpret_cast<T2*>(in));
}


template<class T>
static T get_type(int const type, memptr in)
{
    switch(static_cast<dtype>(type)) {
        case dtype::Int:
            return convert<T, int>(in);
        case dtype::Long:
            return convert<T, long>(in);
        case dtype::Size_t:
            return convert<T, size_t>(in);

        case dtype::Int8:
            return convert<T, int8_t>(in);
        case dtype::Int16:
            return convert<T, int16_t>(in);
        case dtype::Int32:
            return convert<T, int32_t>(in);
        case dtype::Int64:
            return convert<T, int64_t>(in);

        case dtype::UInt8:
            return convert<T, uint8_t>(in);
        case dtype::UInt16:
            return convert<T, uint16_t>(in);
        case dtype::UInt32:
            return convert<T, uint32_t>(in);
        case dtype::UInt64:
            return convert<T, uint64_t>(in);

        case dtype::Float32:
            return convert<T, float>(in);
        case dtype::Float64:
            return convert<T, float>(in);

        //case dtype::Complex64:
            //return convert<T, cpx64>(in);
        //case dtype::Complex128:
            //return convert<T, cpx128>(in);        
    }
}


struct Memory 
{
    Memory(): _memory(nullptr), _size(0) {};
    Memory(idx const size);

    Memory(Memory const&) = default;
    Memory(Memory&&) = default;
    
    Memory& operator=(Memory const&) = default;
    Memory& operator=(Memory&&) = default;
    
    ~Memory() = default;
    
    void alloc(long size);
    
    memptr get() const noexcept { return _memory.get(); }

    long size() const noexcept {
        return _size;
    }

    std::unique_ptr<memtype[]> _memory;
    long _size;
};


// TODO: better converting with std::function

template<class T, idx ndim = Dynamic>
struct Array {
    ArrayInfo& array;
    memptr data;
    ptr<idx> strides;
    std::function<T(memptr)> convert;


    Array() = default;
    
    Array(Array const&) = default;
    Array(Array&&) = default;
    
    Array& operator=(Array const&) = default;
    Array& operator=(Array&&) = default;
    
    ~Array() = default;
    
    /*
    T get(idx const ii)
    {
        return get_type<T>(type, data + ii * strides[0]);
    }


    T const get(idx const ii) const
    {
        return get_type<T>(type, data + ii * strides[0]);
    }
    */
};

template<class T>
using DArray = Array<T, Dynamic> ;


template<class T, idx ndim = Dynamic>
struct View
{
    Memory memory;
    ptr<T> data;
    ptr<idx> _shape, _strides;


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
    int type, is_numpy;
    idx ndim, ndata, datasize;
    ptr<idx> shape, strides;
    memptr data;
    
    ArrayInfo() = delete;
    ~ArrayInfo() = default;
    
    
    /*
    bool check_ndim(idx const ndim) const
    {
        if (ndim != this->ndim) {
            return true;
        }
        return false;
    }
    */

    template<idx ndim>
    void check_dim()
    {
        static_assert(ndim > 0 or ndim == Dynamic,
                      "ndim should be either a positive integer or Dynamic");
        
        auto const _ndim = this->ndim;
        
        if (ndim != Dynamic and ndim != _ndim) {
            printf("View ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }


    template<class T, idx ndim = Dynamic>
    View<T, ndim> view()
    {
        check_dim<ndim>();
        
        auto const& arr_type = type_info(type), req_type = type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("View id: %d, array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        
        check_match(arr_type, req_type);

        View<T, ndim> ret;
        ret._shape = shape;
        
        if (is_numpy) {
            auto const _ndim = this->ndim;
            
            ret.memory.alloc(TypeInfo<idx>::size * _ndim);
            
            ret._strides = reinterpret_cast<idx*>(ret.memory.get());
            idx const dsize = datasize;

            
            for (idx ii = 0; ii < _ndim; ++ii) {
                ret._strides[ii] = idx(double(strides[ii]) / dsize);
            }
        }
        else {
            ret._strides = strides;
        }
        
        ret.data = reinterpret_cast<T*>(data);

        return ret;
    }
    
    
    template<class T, idx ndim = Dynamic>
    Array<T, ndim> array()
    {
        check_dim<ndim>();

        auto const& arr_type = type_info(type), req_type = type_info<T>();

        check_match(arr_type, req_type);
    }
};


// aux namespace
}

#endif
