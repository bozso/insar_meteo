#ifndef __ARRAY_HPP
#define __ARRAY_HPP

#include "numpy/arrayobject.h"

#define type_assert(T, P, msg) static_assert(T<P>::value, msg)


namespace numpy {

// Forward declarations

template<class T>
struct view;



using idx = npy_intp;

template<class T>
using ptr = T*;

template<class T>
using cptr = ptr<T> const;


// Dynamic number of dimensions
static idx constexpr Dynamic = -1;
static idx constexpr row = 0;
static idx constexpr col = 1;

static idx constexpr maxdim = 64;

struct array : PyArrayObject_fields {
    int const ndim() const
    {
        return PyArray_NDIM(this);
    }
    
    void* data() const
    {
        return PyArray_DATA(this);
    }
    
    void enable_flags(int const flags)
    {
        PyArray_ENABLEFLAGS(this, flags);
    }
    
    int const flags() const
    {
        return PyArray_FLAGS(this);
    }
    
    idx const nbytes() const
    {
        return PyArray_NBYTES(this);
    }
    
    //idx const 
    
    
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
            printf("view ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }
    
    
    template<class T>
    view<T> view(idx const ndim) const
    {
        basic_check<T>(ndim());
        
        auto const& arr_type = get_type(), req_type = type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("view id: %d, array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        
        check_match(arr_type, req_type);

        auto ret = view<T>();
        ret._shape = shape();
        ret._strides = {0};
        
        
        auto const _ndim = this->ndim();
        
        idx const dsize = this->datasize();
        
        for (idx ii = 0; ii < _ndim; ++ii) {
            ret._strides[ii] = idx(double(strides[ii]) / dsize);
        }
        
        ret.data = reinterpret_cast<T*>(this->data());
        return ret;
    }    
};



using inarray = array const&;
using outarray = array&;


template<class T>
struct view
{
    // TODO: make appropiate items constant
    ptr<T> data;
    ptr<idx const> _shape;
    std::array<idx, maxdim> _strides;


    view() = default;
    
    view(View const&) = default;
    view(View&&) = default;
    
    view& operator=(View const&) = default;
    view& operator=(View&&) = default;
    
    ~view() = default;
    
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



// end namespace numpy
}

#endif
