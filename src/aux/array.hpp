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
    // taken from pybind11/numpy.h
    
    enum constants {
        NPY_ARRAY_C_CONTIGUOUS_ = 0x0001,
        NPY_ARRAY_F_CONTIGUOUS_ = 0x0002,
        NPY_ARRAY_OWNDATA_ = 0x0004,
        NPY_ARRAY_FORCECAST_ = 0x0010,
        NPY_ARRAY_ENSUREARRAY_ = 0x0040,
        NPY_ARRAY_ALIGNED_ = 0x0100,
        NPY_ARRAY_WRITEABLE_ = 0x0400,
        NPY_BOOL_ = 0,
        NPY_BYTE_, NPY_UBYTE_,
        NPY_SHORT_, NPY_USHORT_,
        NPY_INT_, NPY_UINT_,
        NPY_LONG_, NPY_ULONG_,
        NPY_LONGLONG_, NPY_ULONGLONG_,
        NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
        NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_,
        NPY_OBJECT_ = 17,
        NPY_STRING_, NPY_UNICODE_, NPY_VOID_
    };
    
    
    int const ndim() const { return PyArray_NDIM(this); }
    
    void* data() const { return PyArray_DATA(this); }
    
    void enable_flags(int const flags) { PyArray_ENABLEFLAGS(this, flags); }
    
    int const flags() const { return PyArray_FLAGS(this); }
    idx const nbytes() const { return PyArray_NBYTES(this); }
    
    ptr<idx const> shape() const { return PyArray_SHAPE(this); }
    ptr<idx const> strides() const { return PyArray_STRIDES(this); }
    idx const datasize() const { return PyArray_ITEMSIZE(this); }
    
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
        
        static_assert(
            not std::is_void<T>::value and
            not std::is_null_pointer<T>::value and
            not std::is_pointer<T>::value,
            "Type T should not be void, null pointer or pointer!"
        );
        
        auto const _ndim = this->ndim();
        
        if (ndim != Dynamic and ndim != _ndim) {
            printf("view ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }
    
    
    template<class T>
    view<T> view(idx const ndim) const
    {
        basic_check<T>(ndim);

        auto const _ndim = this->ndim();
        auto const& arr_type = get_type(), req_type = type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("view id: %d, array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        
        check_match(arr_type, req_type);

        auto ret = view<T>();
        ret._shape = this->shape();
        ret._strides = {0};
        
        auto const dsize = this->datasize();
        cptr<idx const> strides = this->strides();
        
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
    ptr<T> data = nullptr;
    ptr<idx const> _shape = nullptr;
    std::array<idx, maxdim> _strides = {0};


    view() = default;
    
    view(view const&) = default;
    view(view&&) = default;
    
    view& operator=(view const&) = default;
    view& operator=(view&&) = default;
    
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

// end guard
#endif
