#ifndef AUX_HPP
#define AUX_HPP

#include <complex>
#include <memory>

#include "type_info.hpp"

namespace aux {

template<class T>
using ptr = T*;

struct Array;
typedef Array* array_ptr;
typedef long idx;

typedef char memtype;
typedef memtype* memptr;

static idx constexpr Dynamic = -1;

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



struct Array
{
    int type, is_numpy;
    idx ndim, ndata, datasize;
    ptr<idx> shape, strides;
    memptr data;
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
};


// aux namespace
}

#endif
