#ifndef __ARRAY_HPP
#define __ARRAY_HPP


#include <array>
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <complex>
#include <unordered_map>

#include <Python.h>


namespace std {

template<class T>
struct is_complex : false_type {};


template<class T>
struct is_complex<complex<T>> : true_type {};

}


namespace aux {

using float32 = float;
using float64 = double;

using cpx64  = std::complex<float32>;
using cpx128 = std::complex<float64>;


using idx = Py_intptr_t;

// Dynamic number of dimensions
static idx constexpr Dynamic = -1;
static idx constexpr row = 0;
static idx constexpr col = 1;
static idx constexpr maxdim = 64;


template<class T>
using ptr = T*;

template<class T>
using cptr = ptr<T> const;


template<class T>
using ref = T&;

template<class T>
using cref = ref<T const>;


using memtype = void;
using memptr = ptr<memtype>;

using name_t = char const*const;

template<class T>
using convert_fun = std::function<T(memptr)>;


enum class dtype {
    Unknown = 0,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Complex64, Complex128
};    


// Forward declarations

struct type_info;

struct Array;

struct PyArrayInterface;
using itf_t = PyArrayInterface;

template<class T>
struct View;

template<class T>
struct ConstView;


template<class from, class to>
static constexpr convert_fun<to> make_convert();

template<class T>
static convert_fun<T> const converter(dtype const id);

template<class T>
static type_info constexpr make_type(name_t name = "Unknown");


template<class T, bool exact_match>
static
std::pair<cref<itf_t>, dtype const>
basic_check(cref<Array> ref, idx const ndim);


template<class T>
static ConstView<T> const const_view(cref<Array> ref, idx const ndim)
{
    auto const pair = basic_check<T, false>(ref, ndim);

    return ConstView<T>(pair.first, pair.second);
}


/*
template<class T>
View<T> Array::view(idx const ndim)
{
    basic_check<T, true>(ndim);
    return View<T>(*this, ndim);
} 
*/


/* Internal structure of PyCapsule */
struct PyCapsule {
    PyObject_HEAD
    void *pointer;
    const char *name;
    void *context;
    PyCapsule_Destructor destructor;
};


struct PyArrayInterface {
  int two;              /* contains the integer 2 -- simple sanity check */
  int nd;               /* number of dimensions */
  char typekind;        /* kind in array --- character code of typestr */
  int itemsize;         /* size of each element */
  int flags;            /* flags indicating how the data should be interpreted */
                        /*   must set ARR_HAS_DESCR bit to validate descr */
  Py_intptr_t *shape;   /* A length-nd array of shape information */
  Py_intptr_t *strides; /* A length-nd array of stride information */
  void *data;           /* A pointer to the first element of the array */
  PyObject *descr;      /* NULL or data-description (same as descr key
                                of __array_interface__) -- must set ARR_HAS_DESCR
                                flag or this will be ignored. */
};


struct Array : PyCapsule {
    Array() = default;
    ~Array() = default;
};


static cref<itf_t> interface(cref<Array> ref)
{
    return *reinterpret_cast<ptr<itf_t>>(ref.pointer);
}


struct ArrayMeta {
    ptr<idx const> shape = nullptr, strides = nullptr;
    idx ndim = 0;
    
    explicit ArrayMeta(cref<itf_t> ref)
    :
    shape{ref.shape}, strides{ref.strides}, ndim{ref.nd} {}
    
    ArrayMeta() = default;
    ~ArrayMeta() = default;
    
    ArrayMeta(ArrayMeta const&) = default;
    ArrayMeta(ArrayMeta&&) = default;
    
    ArrayMeta& operator=(ArrayMeta const&) = default;
    ArrayMeta& operator=(ArrayMeta&&) = default;
    
    
    idx const operator()(idx const ii) const
    {
        return ii * strides[0];
    }

    idx const operator()(idx const ii, idx const jj) const
    {
        return ii * strides[0] + jj * strides[jj];
    }

    idx const operator()(idx const ii, idx const jj, idx const kk) const
    {
        return ii * strides[0] + jj * strides[jj] + kk * strides[kk];
    }

    idx const operator()(idx const ii, idx const jj, idx const kk,
                         idx const ll) const
    {
        return   ii * strides[0] + jj * strides[1] 
               + kk * strides[2] + ll * strides[3];
    }
};


template<class T>
struct ConstView {
    using val_t = T;
    using cval_t = T const;
    using ref_t = ref<T>;
    using cref_t = cref<T>;
    
    memptr data;
    ArrayMeta meta;
    convert_fun<T> const convert;
    
    
    explicit ConstView(cref<itf_t> aref, dtype const id)
    :
    data{aref.data}, meta{aref}, convert{converter<T>(id)}
    {}

    
    ConstView() = delete;
    ~ConstView() = default;

    ConstView(ConstView const&) = default;
    ConstView(ConstView&&) = default;
    
    ConstView& operator=(ConstView const&) = default;
    ConstView& operator=(ConstView&&) = default;
    
    
    template<class... Args>
    val_t operator()(Args&&... args)
    {
        return convert(data + meta(std::forward<Args>(args)...));
    }
    
    template<class... Args>
    cval_t operator()(Args&&... args) const
    {
        return convert(data + meta(std::forward<Args>(args)...));
    }
};


template<class T>
struct id_idx {
    static constexpr dtype value = dtype::Unknown;
};


template<> struct id_idx<int8_t> {
    static constexpr dtype value = dtype::Int8;
};

/*
template<> struct id_idx<int8_t> {
    static constexpr dtype value = dtype::Int8;
};
*/

struct type_info {
    bool const is_pointer = false, is_void = false, is_complex = false,
               is_float = false, is_scalar = false,
               is_arithmetic = false, is_pod = false;
    size_t const size = 0;
    dtype const id = dtype::Unknown;
    name_t name = "Unknown";

    type_info() = default;
    ~type_info() = default;
    
    constexpr type_info(bool const is_pointer, bool const is_void,
                        bool const is_complex, bool const is_float,
                        bool const is_scalar, bool const is_arithmetic,
                        bool const is_pod, size_t const size, dtype const id,
                        name_t name = "Unknown") :
                is_pointer(is_pointer), is_void(is_void),
                is_complex(is_complex), is_float(is_float),
                is_scalar(is_scalar), is_arithmetic(is_arithmetic),
                is_pod(is_pod), size(size), id(id), name(name) {}
    
    bool operator==(cref<type_info> other) const { return id == other.id; }
};



// from: https://www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};


using type_key = std::pair<char, int>;
using type_dict_t = std::unordered_map<type_key const, dtype const, pair_hash>;


static type_dict_t const type_dict{
    {{'b', 1},  dtype::Unknown},    // bool
    {{'S', 1},  dtype::Unknown},    // bytes
    {{'U', 4},  dtype::Unknown},    // string
    
    {{'i', 1},  dtype::Int8},
    {{'i', 2},  dtype::Int16},
    {{'i', 4},  dtype::Int32},
    {{'i', 8},  dtype::Int64},

    {{'u', 1},  dtype::UInt8},
    {{'u', 2},  dtype::UInt16},
    {{'u', 4},  dtype::UInt32},
    {{'u', 8},  dtype::UInt64},
    
    {{'f', 2},  dtype::Unknown},
    {{'f', 4},  dtype::Float32},
    {{'f', 8},  dtype::Float64},

    {{'c', 4},  dtype::Unknown},
    {{'c', 8},  dtype::Complex64},
    {{'c', 16}, dtype::Complex128},

    {{'O', 8}, dtype::Unknown},     //  PyObject, ?
    {{'V', 8}, dtype::Unknown},     //  void
    {{'M', 8}, dtype::Unknown},     //  datetime64, ?
    {{'m', 8}, dtype::Unknown}      //  timedelta64, ?
};


template<class T>
static type_info constexpr make_type(name_t name)
{
    return type_info(std::is_pointer<T>::value, std::is_void<T>::value,
                     std::is_complex<T>::value,
                     std::is_floating_point<T>::value,
                     std::is_scalar<T>::value,
                     std::is_arithmetic<T>::value,
                     std::is_pod<T>::value,
                     sizeof(T), dtype::Unknown, name);
                     
                     // sizeof(T), tpl2dtype<T>(), name);
                     // TODO: fix this
}


static constexpr type_info types[] = {
    [int(dtype::Unknown)] = type_info(),
    
    [int(dtype::Int8)]  = make_type<int8_t>("int8"),
    
    [int(dtype::Int16)] = make_type<int16_t>("int16"),
    [int(dtype::Int32)] = make_type<int32_t>("int32"),
    [int(dtype::Int64)] = make_type<int64_t>("int64"),
    
    [int(dtype::UInt8)]  = make_type<int8_t>("uint8"),
    [int(dtype::UInt16)] = make_type<int16_t>("uint16"),
    [int(dtype::UInt32)] = make_type<int32_t>("uint32"),
    [int(dtype::UInt64)] = make_type<int64_t>("uint64"),
    
    [int(dtype::Float32)] = make_type<float32>("float32"),
    [int(dtype::Float64)] = make_type<float64>("float64"),
    
    [int(dtype::Complex64)]  = make_type<cpx64>("complex64"),
    [int(dtype::Complex128)] = make_type<cpx128>("complex128")
};


//static constexpr dtype types[] = {
    // [int(dtype::Unknown)] = dtype::Unknown,
    /*
    [int(dtype::Int8)]  = make_type<int8_t>("int8"),
    
    [dtype::Int16] = make_type<int16_t>("int16"),
    [dtype::Int32] = make_type<int32_t>("int32"),
    [dtype::Int64] = make_type<int64_t>("int64"),
    
    [dtype::UInt8]  = make_type<int8_t>("uint8"),
    [dtype::UInt16] = make_type<int16_t>("uint16"),
    [dtype::UInt32] = make_type<int32_t>("uint32"),
    [dtype::UInt64] = make_type<int64_t>("uint64"),
    
    [dtype::Float32] = make_type<float32>("float32"),
    [dtype::Float64] = make_type<float64>("float64"),
    
    [dtype::Complex64]  = make_type<cpx64>("complex64"),
    //[dtype::Complex128] = make_type<cpx128>("complex128")
    */
// };


template<class from, class to>
static constexpr convert_fun<to> make_convert()
{
    return [](memptr const in) {
        return static_cast<to>(*reinterpret_cast<from*>(in));
    };
}


// TODO: separate real and complex cases
template<class T>
static convert_fun<T> const converter(dtype const id)
{
    switch(id) {
        case dtype::Int8:
            return make_convert<int8_t, T>();
        case dtype::Int16:
            return make_convert<int16_t, T>();
        case dtype::Int32:
            return make_convert<int32_t, T>();
        case dtype::Int64:
            return make_convert<int64_t, T>();

        case dtype::UInt8:
            return make_convert<uint8_t, T>();
        case dtype::UInt16:
            return make_convert<uint16_t, T>();
        case dtype::UInt32:
            return make_convert<uint32_t, T>();
        case dtype::UInt64:
            return make_convert<uint64_t, T>();

        case dtype::Float32:
            return make_convert<float32, T>();
        case dtype::Float64:
            return make_convert<float64, T>();

        //case dtype::Complex64:
            //return convert<T, cpx64>;
        //case dtype::Complex128:
            //return convert<T, cpx128>;        

        default:
            throw std::runtime_error("AA");
    }
}


cref<type_info> get_type(char const kind, int const size)
{
    return types[int(type_dict.at({kind, size}))];
}

template<class T, bool exact_match>
static
std::pair<cref<itf_t>, dtype const>
basic_check(cref<Array> ref, idx const ndim)
{
    auto const& itf = interface(ref);
    auto const  nd = itf.nd;
    
    auto const& arr_type = get_type(itf.typekind, itf.itemsize);
    auto const  req_type = make_type<T>();
    
    if (arr_type.is_complex != req_type.is_complex) {
        throw std::runtime_error("Cannot convert complex to non complex "
                                 "type!");
    }
    
    if (exact_match and arr_type.id != req_type.id) {
        printf("View id: %d, Array id: %d\n",
                static_cast<int>(arr_type.id),
                static_cast<int>(req_type.id)); 
        throw std::runtime_error("Not same id!");
    }
    
    if (ndim < 0 and ndim != Dynamic and ndim < maxdim) {
        throw std::runtime_error("ndim should be either a "
                                 "positive integer that is less than"
                                 "the maximum number of dimensions or "
                                 "aux::Dynamic");
    }
    
    
    static_assert(
        not std::is_void<T>::value and
        not std::is_pointer<T>::value,
        "Type T should not be void, a null pointer or a pointer!"
    );
    
    if (ndim != nd) {
        printf("view ndim: %ld, array ndim: %d\n", ndim, nd); 
        throw std::runtime_error("Dimension mismatch!");
    }
    
    return {itf, arr_type.id};
}


/*
template<class T>
struct View
{
    using val_t = T;
    using cval_t = T const;
    using ref_t = ref<T>;
    using cref_t = cref<T>;
    
    // TODO: make appropiate items constant
    cptr<T> data;
    std::array<idx, maxdim> _strides;
    ArrayMeta meta;

    explicit View(ref<Array> ref, idx const ndim)
    :
    data{reinterpret_cast<T*>(ref.data)}
    {
        auto const& itf = ref.interface()
        meta.shape = ref.shape;
        meta.ndim = ref.ndim;
        
        for (idx ii = 0; ii < ndim; ++ii) {
            _strides[ii] = idx(double(ref.strides[ii]) / ref.datasize);
        }
        
        meta.strides = _strides.data();
    }
    
    
    View() = delete;
    ~View() = default;
    
    View(View const&) = default;
    View(View&&) = default;
    
    View& operator=(View const&) = default;
    View& operator=(View&&) = default;
    
    
    cref<idx> shape(idx const ii) const noexcept
    {
        return meta.shape[ii];
    }
    
    
    template<class... Args>
    ref_t operator()(Args&&... args)
    {
        return data[meta(std::forward<Args>(args)...)];
    }
    
    
    template<class... Args>
    cref_t operator()(Args&&... args) const
    {
        return data[meta(std::forward<Args>(args)...)];
    }
};
*/


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


// end namespace aux
}

// end guard
#endif
