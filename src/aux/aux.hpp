#ifndef AUX_HPP
#define AUX_HPP

#include <iostream>
#include <complex>
#include <memory>

#include "type_info.hpp"

namespace aux {

// from https://www.modernescpp.com/index.php/c-core-guidelines-rules-for-variadic-templates

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



template<class T>
using ptr = T*;

typedef long idx;
static idx constexpr Dynamic = -1;

struct ArrayMeta;
typedef ptr<ArrayMeta> array_ptr;


template<idx ndim>
struct Array;

typedef Array<Dynamic> DArray;

typedef char memtype;
typedef ptr<memtype> memptr;



RTypeInfo const& type_info(int const type);
RTypeInfo const& type_info(dtype const type);


template<class T>
RTypeInfo const& type_info()
{
    return type_info(tpl2dtype<T>());
}


enum layout {
    colmajor,
    rowmajor
};



struct ArrayMeta {
    int type, is_numpy;
    idx ndim, ndata, datasize;
    ptr<idx> shape, strides;
    memptr data;
};


template<class T1, class T2>
static T1 convert(memptr in)
{
    return static_cast<T1>(*reinterpret_cast<T2*>(in));
}


template<class T>
static T get_type(dtype type, memptr in)
{
    switch(type) {
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



// TODO: better converting with std::function

template<idx ndim>
struct Array {
    Array() = delete;
    ~Array() = default;
    
    Array(array_ptr const arr) : _arr(*arr),
                                 type(static_cast<dtype>(arr->type)),
                                 data(arr->data), strides(arr->strides)
    {
        static_assert(ndim > 0 or ndim == Dynamic,
                      "ndim should be a positive integer or Dynamic");
        
        if (ndim != Dynamic and ndim != arr->ndim) {
            print("View ndim: %, array ndim: %\n", ndim, arr->ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }
    
    
    template<class T>
    T get(idx const ii)
    {
        return get_type<T>(type, data + ii * strides[0]);
    }

    template<class T>
    T const get(idx const ii) const
    {
        return get_type<T>(type, data + ii * strides[0]);
    }

    idx const& shape(idx const ii) const { return _arr.shape[ii]; }


private:
    ArrayMeta& _arr;
    dtype type;
    memptr data;
    ptr<idx> strides;
};

struct Memory 
{
    Memory(): memory(nullptr), _size(0) {};
    ~Memory() = default;
    
    Memory(long size);
    void alloc(long size);
    
    memptr get() const noexcept { return memory.get(); }

    template<class T>
    ptr<T> ptr(idx const offset = 0) const noexcept {
        return reinterpret_cast<T*>(memory.get());
    }
    
    long size() const noexcept {
        return this->_size;
    }

    std::unique_ptr<memtype[]> memory;
    long _size;
};



template<class T, idx ndim = Dynamic>
struct View
{
private:
    Memory memory;
    ptr<T> data;
    ptr<idx> _shape, strides;

public:
    View() = default;
    ~View() = default;
    
    View(array_ptr const arr) : memory(), _shape(arr->shape)
    {
        static_assert(ndim > 0 or ndim == Dynamic,
                      "ndim should be a positive integer or Dynamic");
        
        idx const _ndim = arr->ndim;
        
        if (ndim != Dynamic and ndim != _ndim) {
            printf("View ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }

        
        auto const& arr_type = type_info(arr->type), req_type = type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("View id: %d, array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        

        if (arr_type.is_complex and not req_type.is_complex) {
            throw std::runtime_error("Input array is complex but not "
                                     "complex array is requested!");
        }

        if (not arr_type.is_complex and req_type.is_complex) {
            throw std::runtime_error("Input array is not complex but "
                                     "complex array is requested!");
        }


        idx const dsize = arr->datasize;
        
        if (arr->is_numpy) {
            memory.alloc(TypeInfo<idx>::size * _ndim);
            strides = memory.ptr<idx>();
            
            for (idx ii = 0; ii < _ndim; ++ii) {
                strides[ii] = idx( double(arr->strides[ii]) / dsize );
            }
        }
        else {
            strides = arr->strides;
        }
        
        data = reinterpret_cast<T*>(arr->data);
    }
    
    idx const& shape(idx const ii) const { return _shape[ii]; }
    
    T& operator()(idx const ii)
    {
        return data[ii * strides[0]];
    }

    T& operator()(idx const ii, idx const jj)
    {
        return data[ii * strides[0] + jj * strides[1]];
    }

    T& operator()(idx const ii, idx const jj, idx const kk)
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }

    T& operator()(idx const ii, idx const jj, idx const kk, idx const ll)
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[4]];
    }


    T const& operator()(idx const ii) const
    {
        return data[ii * strides[0]];
    }

    T const& operator()(idx const ii, idx const jj) const
    {
        return data[ii * strides[0] + jj * strides[1]];
    }

    T const& operator()(idx const ii, idx const jj, idx const kk) const
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]];
    }

    T const& operator()(idx const ii, idx const jj, idx const kk, idx const ll) const
    {
        return data[ii * strides[0] + jj * strides[1] + kk * strides[2]
                    + ll * strides[4]];
    }
};


// aux namespace
}

#endif
