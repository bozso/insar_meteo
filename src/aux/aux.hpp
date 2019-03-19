#ifndef AUX_HPP
#define AUX_HPP

#include <complex>
#include <memory>

#include "type_info.hpp"

namespace aux {

struct Array;
typedef Array* array_ptr;
typedef long idx;

typedef char memtype;
typedef memtype* memptr;


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
    idx ndim, ndata, datasize, *shape, *strides;
    memptr data;
};


struct Memory 
{
    Memory(): memory(nullptr), _size(0) {};
    ~Memory() = default;
    
    Memory(long size);
    void alloc(long size);
    
    memptr get() const noexcept { return this->memory.get(); }
    
    long size() const noexcept {
        return this->_size;
    }

    std::unique_ptr<memtype[]> memory;
    long _size;
};


#define convert(T1, T2)                                                   \
do {                                                                      \
    auto adata =  reinterpret_cast<T2*>(arr->data);                       \
    for (idx ii = 0; ii < ndata; ++ii) {                                  \
        data[ii] = static_cast<T1>(*(adata + ii));                        \
    }                                                                     \
} while(0)



/*
template<class T, class TI = TypeInfo<T> >
struct View
{
private:
    Memory memory;
    T* data;
    idx ndim, *shape, *strides;

public:
    View() = default;
    ~View() = default;
    
    View(array_ptr arr) : ndim(arr->ndim), shape(arr->shape)
    {
        int arr_type = arr->type, req_type = TI::id;
        bool arr_cpx = is_complex(arr_type), req_cpx = TI::is_complex;
        bool cast = arr_type != req_type, is_np = arr->is_numpy;
        
        idx dsize = arr->datasize, ndata = arr->ndata;
        long size = 0;
        constexpr long size_strides = sizeof(idx) * ndim;
        
        if (is_np) {
            size += size_strides;
        }
        
        if (cast) {
            size += dsize * arr->ndata;
        }
        
        memory.alloc(size);
        
        if (is_np) {
            strides = memory.get();
            
            for (idx ii = 0; ii < ndim; ++ii) {
                strides[ii] = idx( double(arr->strides) / dsize );
            }
        }
        else {
            strides = arr->strides;
        }
        
        
        if (arr_cpx and not req_cpx) {
            throw std::runtime_error("Input array is complex but not "
                                     "complex array is requested!");
        }

        if (not arr_cpx and req_cpx) {
            throw std::runtime_error("Input array is not complex but "
                                     "complex array is requested!");
        }
        
        if (cast) {
            // cast
            if (is_np) {
                data = reinterpret_cast<T*>(memory.get() + size_strides);
                
                switch(arr_type) {
                    case 1:
                        convert(T, int8_t);
                        break;
                    case 2:
                        convert(T, int);
                        break;
                    case 3:
                        convert(T, long);
                        break;
                    case 4:
                        convert(T, size_t);
                        break;

                    case 5:
                        convert(T, int8_t);
                        break;
                    case 6:
                        convert(T, int16_t);
                        break;
                    case 7:
                        convert(T, int32_t);
                        break;
                    case 8:
                        convert(T, int64_t);
                        break;

                    case 9:
                        convert(T, uint8_t);
                        break;
                    case 10:
                        convert(T, uint16_t);
                        break;
                    case 11:
                        convert(T, uint32_t);
                        break;
                    case 12:
                        convert(T, uint64_t);
                        break;

                    case 13:
                        convert(T, float);
                        break;
                    case 14:
                        convert(T, double);
                        break;

                    case 15:
                        convert(T, cpx64);
                        break;
                    case 16:
                        convert(T, cpx128);
                        break;
                }
            }
            else {
                // nocast
                data = reinterpret_cast<T*>(arr->data);
            }
        }
    }
};
*/


// aux namespace
}

#endif
