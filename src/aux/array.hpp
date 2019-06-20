#ifndef __ARRAY_HPP
#define __ARRAY_HPP


#include <array>
#include <functional>
#include <type_traits>
#include <stdexcept>

#include "aux.hpp"
#include "type_info.hpp"


namespace numpy {

// Forward declarations

template<class T>
struct View;


template<class T>
struct ConstView;


using idx = long;


// Dynamic number of dimensions
static idx constexpr Dynamic = -1;
static idx constexpr row = 0;
static idx constexpr col = 1;
static idx constexpr maxdim = 64;


static void check_match(aux::RTypeInfo const& one, aux::RTypeInfo const& two)
{
    if (one.is_complex != two.is_complex) {
        throw std::runtime_error("Cannot convert complex to non complex "
                                 "type!");
    }
}


// taken from pybind11/numpy.h

enum flags {
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


struct Array {
    int const type = 0, is_numpy = 0;
    idx const ndim = 0, ndata = 0, datasize = 0;
    aux::cptr<idx const> shape = nullptr, strides = nullptr;
    aux::memptr data = nullptr;
    
    
    aux::RTypeInfo const& get_type() const
    {
        return aux::type_info(type);
    }
    
    
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
            not std::is_pointer<T>::value,
            "Type T should not be void, a null pointer or a pointer!"
        );
        
        auto const _ndim = this->ndim;
        
        if (ndim != Dynamic and ndim != _ndim) {
            printf("view ndim: %ld, array ndim: %ld\n", ndim, _ndim); 
            throw std::runtime_error("Dimension mismatch!");
        }
    }
    
    
    template<class T>
    ConstView<T> const_view(idx const ndim) const
    {
        basic_check<T>(ndim);

        auto const& arr_type = get_type(), req_type = aux::type_info<T>();
        check_match(arr_type, req_type);

        return ConstView<T>(*this, ndim);
    }
    
    
    template<class T>
    View<T> view(idx const ndim)
    {
        basic_check<T>(ndim);

        auto const& arr_type = get_type(), req_type = aux::type_info<T>();
        
        if (arr_type.id != req_type.id) {
            printf("View id: %d, Array id: %d\n", arr_type.id, req_type.id); 
            throw std::runtime_error("Not same id!");
        }
        
        check_match(arr_type, req_type);

        return View<T>(*this, ndim);
    }    
};


using arr_in = Array const&;
using arr_out = Array&;


template<class T>
struct View
{
    // TODO: make appropiate items constant
    aux::ptr<T> data = nullptr;
    aux::ptr<idx const> _shape = nullptr;
    std::array<idx, maxdim> _strides{0};


    View() = default;
    
    explicit View(Array& ref, idx const ndim)
    :
    _shape(ref.shape), data(reinterpret_cast<T*>(ref.data))
    {
        for (idx ii = 0; ii < ndim; ++ii) {
            _strides[ii] = idx(double(ref.strides[ii]) / ref.datasize);
        }
        
    }
    
    
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



template<class T>
struct ConstView {
    using value_type = T;
    using convert_fun = std::function<value_type(aux::memptr)>;
    
    
    Array const& ref;
    aux::memptr const data = nullptr;
    aux::ptr<idx const> const strides;
    convert_fun const convert;
    
    
    explicit ConstView(Array const& ref, idx const ndim)
    :
    ref(ref), data(ref.data), strides(ref.strides), convert(factory(ref.type))
    {}

    
    ConstView() = delete;
    ~ConstView() = default;

    ConstView(ConstView const&) = default;
    ConstView(ConstView&&) = default;
    
    ConstView& operator=(ConstView const&) = default;
    ConstView& operator=(ConstView&&) = default;
    
    
    template<class P>
    static constexpr convert_fun make_convert()
    {
        return [](aux::memptr const in) {
            return static_cast<T>(*reinterpret_cast<P*>(in));
        };
    }

    
    // TODO: separate real and complex cases
    static convert_fun const factory(int const type)
    {
        switch(static_cast<aux::dtype>(type)) {
            case aux::dtype::Int:
                return make_convert<int>();
            case aux::dtype::Long:
                return make_convert<long>();
            case aux::dtype::Size_t:
                return make_convert<size_t>();
    
            case aux::dtype::Int8:
                return make_convert<int8_t>();
            case aux::dtype::Int16:
                return make_convert<int16_t>();
            case aux::dtype::Int32:
                return make_convert<int32_t>();
            case aux::dtype::Int64:
                return make_convert<int64_t>();

            case aux::dtype::UInt8:
                return make_convert<uint8_t>();
            case aux::dtype::UInt16:
                return make_convert<uint16_t>();
            case aux::dtype::UInt32:
                return make_convert<uint32_t>();
            case aux::dtype::UInt64:
                return make_convert<uint64_t>();
    
            case aux::dtype::Float32:
                return make_convert<float>();
            case aux::dtype::Float64:
                return make_convert<double>();

            //case dtype::Complex64:
                //return convert<T, cpx64>;
            //case dtype::Complex128:
                //return convert<T, cpx128>;        

            default:
                throw std::runtime_error("AA");
        }
    }
    
    T const operator ()(idx const ii) const
    {
        return convert(data + ii * strides[0]);
    }
};


// end namespace numpy
}

// end guard
#endif
