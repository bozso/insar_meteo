# C++ code

## Numpy to Eigen
```c++
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
class ptr {
    private T* pointer;
    
    typedef T val_t;
    typedef T* ptr_t;
    
    
    ptr(): pointer(nullptr) {}
    ~ptr() = default;

    ptr(ptr const&) = default;
    ptr& operator=(ptr const&) = default;

    ptr(ptr&&) = delete;
    ptr& operator=(ptr&&) = delete;
    
    ptr(ptr_t other) pointer(other) {}
    ptr& operator=(ptr_t const other) { pointer = other };
    
    val_t& operator*() { return *pointer; }
    val_t& operator[](idx ii) { return pointer[ii]; }
    
    val_t const& operator*() const { return *pointer; }
    val_t const& operator[](idx ii) const { return pointer[ii]; }
    
    ptr_t get() const { return pointer; }
};


template<class trait, class T>
void type_assert()
{
    static_assert(trait<T>::value);
}

using DStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template<class T>
using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;



template<class T1, class T2>
DMatrix<T1> convert(array_ptr arr)
{
    auto dsize = arr->datasize;
    auto stride = DStride(arr->strides[0] / dsize, arr->strides[1] / dsize);
    
    auto data = reinterpret_cast<T2*>(arr->data);
    auto tmp = \
    Eigen::Map<DMatrix<T2>, Eigen::Unaligned, DStride>(data, arr->shape[0],
                                                             arr->shape[1],
                                                             stride);
    
    return tmp.template cast <T1>();
}



template<class T>
DMatrix<T> from_numpy(array_ptr arr, bool colvec = true)
{
    assert(arr->ndim == 2);
    
    switch(arr->type)
    {
        case 1:
            return convert<T, bool>(arr);
        case 2:
            return convert<T, int>(arr);
        case 3:
            return convert<T, long>(arr);
        case 4:
            return convert<T, size_t>(arr);

        case 5:
            return convert<T, int8_t>(arr);
        case 6:
            return convert<T, int16_t>(arr);
        case 7:
            return convert<T, int32_t>(arr);
        case 8:
            return convert<T, int64_t>(arr);

        case 9:
            return convert<T, uint8_t>(arr);
        case 10:
            return convert<T, uint16_t>(arr);
        case 11:
            return convert<T, uint32_t>(arr);
        case 12:
            return convert<T, uint64_t>(arr);

        case 13:
            return convert<T, float>(arr);
        case 14:
            return convert<T, double>(arr);

        //case 15:
            //return convert<T, cpx64>(arr);
        //case 16:
            //return convert<T, cpx128>(arr);
    }
}
```

## Lab C++ code 

### lab.hpp

```c++
#ifndef LAB_HPP
#define LAB_HPP


#include <chrono>
#include <algorithm>
#include <string>
#include <complex>
#include <memory>
#include <fstream>
#include <functional>

#include <stdarg.h>


typedef std::fstream::char_type memtype;
typedef memtype* memptr;



// make_unique from https://herbsutter.com/gotw/_102/

template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}




struct Memory {
    Memory(): memory(nullptr), _size(0) {};
    ~Memory() = default;
    
    Memory(long size): _size(size)
    {
        this->memory = make_array<memtype>(size);
    }
    
    void alloc(long size)
    {
        this->_size = size;
        this->memory = make_array<memtype>(size);
    }
    
    memptr get() const noexcept
    {
        return this->memory.get();
    }
    
    template<class T>
    T* offset(long offset) const
    {
        return reinterpret_cast<T*>(this->memory.get() + offset);
    }
    
    long size() const noexcept
    {
        return this->_size;
    }

    std::unique_ptr<memtype[]> memory;
    long _size;
};


typedef std::function<memptr(memptr, long)> swap_fun;


typedef std::complex<double> cpx128;
typedef std::complex<float> cpx64;

enum class dtype : int {
    Unknown     = 0,
    Bool        = 1,
    Int         = 2,
    Long        = 3,
    Size_t      = 4,
    Int8        = 5,
    Int16       = 6,
    Int32       = 7,
    Int64       = 8,
    UInt8       = 9,
    UInt16      = 10,
    UInt32      = 11,
    UInt64      = 12,
    Float32     = 13,
    Float64     = 14,
    Complex64   = 15,
    Complex128  = 16
};


enum class ftype : int {
    Unknown = 0,
    Array   = 1,
    Records = 2
};


// TODO: separate DataFile into DataFile (C++) and FileInfo (C, Python interface)

typedef long idx;


struct FileInfo {
    char *path;
    idx *offsets, ntypes, recsize;
    int *dtypes, filetype, endswap;
};


struct DataFile {
    typedef std::ios_base::openmode iomode;
    
    FileInfo& info;
    Memory mem;
    std::fstream file;
    memptr buffer;
    idx* offsets;
    long nio;
    swap_fun sfun;

    DataFile() = default;
    DataFile(FileInfo* _info, iomode mode);
    void open(iomode mode);    
    
    ~DataFile() = default;


    static std::string const dt2str(int type) noexcept;
    static std::string const ft2str(int type) noexcept;
    
    static dtype str2dt(std::string const& type) noexcept;
    static ftype str2ft(std::string const& type) noexcept;


    template<class T1, class T2>
    static T1 convert(memptr in)
    {
        return static_cast<T1>(*reinterpret_cast<T2*>(in));
    }

    void read_rec();

    template<class T>
    void write_rec(T* obj)
    {
        file.write(reinterpret_cast<memptr>(obj), sizeof(T));
        nio++;
    }
};


class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;
	
	std::chrono::time_point<clock_t> m_beg;
 
public:
	Timer() : m_beg(clock_t::now()) {}
	
	void reset();
	double elapsed() const;
    void report() const;
};


#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)

/**************
 * for macros *
 **************/

#define m_for(ii, max) for(long (ii) = (max); (ii)--; )
#define m_forz(ii, max) for(long (ii) = 0; (ii) < max; ++(ii))
#define m_fors(ii, min, max, step) for(long (ii) = (min); (ii) < (max); (ii) += (step))
#define m_for1(ii, min, max) for(long (ii) = (min); (ii) < (max); ++(ii))


#if defined(_MSC_VER)
        #define m_inline __inline
#elif defined(__GNUC__)
    #if defined(__STRICT_ANSI__)
         #define m_inline __inline__
    #else
         #define m_inline inline
    #endif
#else
    #define m_inline
#endif


#if defined(_OS_WINDOWS_) && defined(_COMPILER_INTEL_)
#  define m_static_inline static
#elif defined(_OS_WINDOWS_) && defined(_COMPILER_MICROSOFT_)
#  define m_static_inline static __inline
#else
#  define m_static_inline static inline
#endif


#if defined(_OS_WINDOWS_) && !defined(_COMPILER_MINGW_)
#  define m_noinline __declspec(noinline)
#  define m_noinline_decl(f) __declspec(noinline) f
#else
#  define m_noinline __attribute__((noinline))
#  define m_noinline_decl(f) f __attribute__((noinline))
#endif


extern "C" {
    
int is_big_endian() noexcept;
long dtype_size(long type) noexcept;

}


struct Number {
    dtype type;
    union {
        bool   b;
        long   il;
        int    ii;
        size_t is;
    
        int8_t  i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
    
        uint8_t  ui8;
        uint16_t ui16;
        uint32_t ui32;
        uint64_t ui64;
    
        float  fl32;
        double fl64;
    
        complex<float>  c64;
        complex<double> c128;
    };
    
    Number() = delete;
    Number(bool n)            : b(n) {};
    Number(int  n)            : ii(n) {};
    Number(long n)            : il(n) {};
    Number(size_t n)          : is(n) {};
    Number(int8_t n)          : i8(n) {};
    Number(int16_t n)         : i16(n) {};
    Number(int32_t n)         : i32(n) {};
    Number(int64_t n)         : i64(n) {};

    Number(uint8_t n)         : ui8(n) {};
    Number(uint16_t n)        : ui16(n) {};
    Number(uint32_t n)        : ui32(n) {};
    Number(uint64_t n)        : ui64(n) {};
    
    Number(float n)           : fl32(n) {};
    Number(double n)          : fl64(n) {};
    
    Number(cpx64 n)  : c64(n) {};
    Number(cpx128 n) : c128(n) {};
    ~Number() = default;
}; 


#define m_max_error_len 256

class error : public std::exception {
public:
    error(std::string&, ...);
    error(std::string&&, ...);
    virtual const char * what() const throw() { return error_msg; }
    virtual ~error() throw() {};
private:
    char error_msg[m_max_error_len];
};


// guard
#endif

```

### lab.cpp

```c++
#include <string>
#include <lab/lab.hpp>

using std::string;

typedef DataFile DT;

error::error(string& format,  ...) : std::exception()
{
    va_list args;
    va_start(args, format);
    vsnprintf(error_msg, m_max_error_len, format.c_str(), args);
    va_end(args);
}

error::error(string&& format, ...) : std::exception()
{
    va_list args;
    va_start(args, format);
    vsnprintf(error_msg, m_max_error_len, format.c_str(), args);
    va_end(args);
}

memptr endswap(memptr objp, long size)
{
    std::reverse(objp, objp + size);
    return objp;
}


memptr noswap(memptr objp, long size)
{
    return objp;
}

DT::DataFile(FileInfo* _info, iomode mode) : info(*_info)
{
    open(mode);
}

void DT::open(iomode mode)
{
    file.open(info.path, mode);
    
    if (not file.is_open())
    {
        throw error("Could not open file: %s!", info.path);
    }
    
    if (info.endswap)
    {
        sfun = endswap;
    }
    else
    {
        sfun = noswap;
    }
    
    mem.alloc(sizeof(memtype) * info.recsize);
    buffer = mem.get();
}


void DataFile::read_rec()
{
    file.read(reinterpret_cast<memptr>(buffer), info.recsize);
    nio++;
}


constexpr int ndtype = 17;

static string const dtype_names[17] = {
    "Unknown",
    "Bool",
    "Int",
    "Long",
    "Size_t",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Complex64",
    "Complex128"
};


string const DT::dt2str(int type) noexcept
{
    return type < ndtype ? dtype_names[type] : dtype_names[0];
}

dtype DT::str2dt(string const& type) noexcept
{
    for (int ii = 0; ii < ndtype; ++ii)
    {
        if (type == dtype_names[ii])
        {
            return static_cast<dtype>(ii);
        }
    }
    
    return dtype::Unknown;
}


static constexpr int nftype = 3;

static string const ftype_names[nftype] = {
    "Unknown",
    "Array",
    "Records"
};


string const DataFile::ft2str(int type) noexcept
{
    return type < nftype ? ftype_names[type] : ftype_names[0];
}

ftype DT::str2ft(string const& type) noexcept
{
    for (int ii = 0; ii < nftype; ++ii)
    {
        if (type == ftype_names[ii])
        {
            return static_cast<ftype>(ii);
        }
    }
    
    return ftype::Unknown;
}


static long const sizes[ndtype] = {
    0,
    sizeof(bool),
    sizeof(int),
    sizeof(long),
    sizeof(size_t),
    sizeof(int8_t),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(int64_t),
    sizeof(uint8_t),
    sizeof(uint16_t),
    sizeof(uint32_t),
    sizeof(uint64_t),
    sizeof(float),
    sizeof(double),
    sizeof(cpx64),
    sizeof(cpx128)
};


extern "C" {

long dtype_size(long type) noexcept
{
    return type < ndtype ? sizes[type] : sizes[0];
}

}

#include "lab/lab.hpp"

void Timer::reset()
{
    this->m_beg = clock_t::now();
}


double Timer::elapsed() const
{
    return std::chrono::duration_cast<second_t>(clock_t::now() - this->m_beg).count();
}


void Timer::report() const
{
    printf("Elapsed time: %lf seconds,\n", this->elapsed());
}


extern "C" {

int is_big_endian() noexcept
{
    short word = 0x4321;
    if ((*(char *)& word) != 0x21 )
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

}
```

# Python code

### lab.py

```python
import os

from tempfile import _get_default_tempdir, _get_candidate_names
from distutils.ccompiler import new_compiler
from os.path import dirname, realpath, join, isfile
from itertools import accumulate

from ctypes import *

filedir = dirname(realpath(__file__))
tmpdir = _get_default_tempdir()


def get_library(libname, searchdir):
    lib_filename = new_compiler().library_filename
    
    libpath = join(searchdir, lib_filename(libname, lib_type="shared"))

    return CDLL(libpath)


lb = get_library("lab", join(filedir, "..", "src", "build"))
im = get_library("inmet", join(filedir, "..", "src", "build"))

lb.is_big_endian.restype = c_int
lb.is_big_endian.argtype = None

lb.dtype_size.restype = c_long
lb.dtype_size.argtype = c_long

big_end = c_int(lb.is_big_endian())

memptr = POINTER(c_char)
c_idx = c_long
c_idx_p = POINTER(c_long)
c_int_p = POINTER(c_int)


class _FileInfo(Structure):
    _fields_ = (
        ("path", c_char_p),
        ("offsets", c_idx_p),
        ("ntypes", c_idx),
        ("recsize", c_idx),
        ("dtypes", c_int_p),
        ("filetype", c_int),
        ("endswap", c_int)
    )


class FileInfo(object):
    filetype2int = {
        "Unknown" : 0,
        "Array" : 1,
        "Records" : 2
    }

    dtype2int = {
        "Unknown"    : 0,
        "Bool"       : 1,
        "Int"        : 2,
        "Long"       : 3,
        "Size_t"     : 4,
        "Int8"       : 5,
        "Int16"      : 6,
        "Int32"      : 7,
        "Int64"      : 8,
        "UInt8"      : 9,
        "UInt16"     : 10,
        "UInt32"     : 11,
        "UInt64"     : 12,
        "Float32"    : 13,
        "Float64"    : 14,
        "Complex64"  : 15,
        "Complex128" : 16
    }

    def __init__(self, filetype, dtypes, path=None, keep=False,
                 endian="native"):

        self.keep, self.info = keep, None
        ntypes = len(dtypes)
        
        if filetype is "Array":
            assert ntypes == 1, "Array one dtype needed!"

        if path is None:
            _path = join(tmpdir, next(_get_candidate_names()))
            path = bytes(_path, "ascii")
        
        filetype = FileInfo.filetype2int[filetype]
        dtypes = (c_int * ntypes)(*(FileInfo.dtype2int[elem] for elem in dtypes))
        
        sizes = tuple(lb.dtype_size(dtypes[ii]) for ii in range(ntypes))
        
        offsets = (c_idx * (ntypes))(0, *accumulate(sizes[:ntypes - 1]))

        
        if endian == "native":
            swap = 0
        elif endian == "big":
            if big_end:
                swap = 0
            else:
                swap = 1
        elif endian == "little":
            if big_end:
                swap = 1
            else:
                swap = 0
        else:
            raise ValueError('endian should either be "big", "little" '
                             'or "native"')
        
        
        self.info = _FileInfo(path, offsets, ntypes, sum(sizes),
                              dtypes, filetype, swap)

        
    def ptr(self):
        return byref(self.info)    

        
    def __del__(self):
        if not self.keep and self.info is not None \
        and isfile(self.info.path):
            os.remove(self.info.path.decode("ascii"))
        
        

def open(name):
    path = bytes(workspace.get(name, "path"), "ascii")
    ntypes = workspace.getint(name, "ntypes")
    recsize = workspace.getint(name, "recsize")
    filetype = workspace.get(name, "filetype")
    endian = workspace.get(name, "endian")


def save(info, name, path):
    pass


def main():
    a = FileInfo("Records", ["Float32", "Float32", "Float32", "Complex128"])
    
    im.test(a.ptr())
    
    
    return 0


if __name__ == "__main__":
    main()
```


# Julia code

## PolyFit.jl

```julia
__precompile__(true)

module PolynomFit

export PolyFit, MultiPolyFit, Scale, fit_poly

struct Scale{T<:Number}
    min::T
    scale::T
    
    Scale(m::Tuple{T,T}) where T<:Number = new{T}(m[1], m[2] - m[1])
end

function Base.show(io::IO, p::Scale{T}) where T<:Number
    print(io, "Scale{$T} Min: $(p.min); Scale: $(p.scale)")
end


struct PolyFit{T<:Number}
    coeffs::Vector{T}
    deg::Int64
    scaled::Bool
    xs::Union{Scale{T}, Nothing}
    ys::Union{Scale{T}, Nothing}
end

function Base.show(io::IO, p::PolyFit{T}) where T<:Number
    print(io, "PolyFit{$T} Fit degree: $(p.deg); Fitted Coefficients: ",
              "$(p.coeffs)")
    
    if p.scaled
        print(io, "Scaled: true; Scale: $(p.scale)\n")
    else
        print(io, "Scaled: false")
    end
end



struct MultiPolyFit{T<:Number}
    fits::Vector{PolyFit{T}}
    nfits::Int64
end


function Base.show(io::IO, p::MultiPolyFit{T}) where T<:Number
    print(io, "MultiPolyFit{$T} Fits: $(p.fits) Number of fits: $(p.nfits)")
end


function fit_poly(x::Vector{T},
                  y::Vector{T},
                  deg::Int64,
                  scaled::Bool=true) where T<:Number

    if scaled
        ny = length(y)
        search = hcat(x, y)
        
        minmax = extrema(search, dims=1)
        
        scales = vec([Scale(m) for m in minmax])
        
        # f(x) = (x - x_min) / (x_max - x_min)
        
        xs = scales[1]
        design = collect( ((xx - xs.min) / xs.scale)^p for xx in x, p in deg:-1:0)
        
        ys = scales[2]
        
        yy = Vector{T}(undef, ny)
        
        @simd for jj in 1:ny
            yy[jj] = (y[jj] - ys.min) / ys.scale
        end
    else    
        xs, ys, design, yy = nothing, nothing, \
        collect(xx^p for xx in x, p in deg:-1:0), y
    end
    
    return PolyFit{T}(design \ yy, deg, scaled, xs, ys)
end


function fit_poly(x::Vector{T},
                  y::Matrix{T},
                  deg::Int64,
                  scaled::Bool=true,
                  dim::Int64=1) where T<:Number
    
    # TODO: make it general for y::Array{T,N}; transpose when dim == 1
    dim <= 0 && error("dim should be > 0!")
    
    n = ndims(y)
    
    @inbounds begin
    
    if n == 1
        dim > 1 && error("")
        yy = y
    elseif n == 2
        if dim == 1
            yy = y
        elseif dim == 2
            yy = transpose(y)
        else
            error("dim >= 2")
        end
    else
        error("Max 2 dim.")
    end

    nfit = size(yy, 2)

    if scaled
        search = hcat(x, yy)
        
        minmax = extrema(search, dims=1)
        
        scales = vec([Scale(m) for m in minmax])
        
        # f(x) = (x - x_min) / (x_max - x_min)
        
        xs = scales[1]
        design = collect( ((xx - xs.min) / xs.scale)^p for xx in x, p in deg:-1:0)
        
        if n == 1
            ys = scales[2]
            @simd for jj in eachindex(yy)
                yy[jj] = (yy[jj] - ys.min) / ys.scale
            end
        else
            rows, cols = size(yy)
            for jj in 1:cols
                ys = scales[jj + 1]
                @simd for ii in 1:rows
                    iscale = 1.0 / ys.scale
                    yy[ii,jj] = (yy[ii,jj] - ys.min) * iscale
                end
            end
        end
    else
        yy, xs, ys = y, nothing, nothing
        design = collect(xx^p for xx in x, p in deg:-1:0)
    end
    
    # @inbounds
    end
    
    return PolyFit{T}(design \ yy, deg, nfit, scaled, scales)
end



#=
struct PolyFit{T<:Number}
    coeffs::VecOrMat{T}
    deg::Int64
    nfit::Int64
    scaled::Bool
    scales::Union{Vector{Scale{T}}, Nothing}

    """
    scale = x_max - x_min
    
    scaled(x)      = (x - x_min) / scale
    scaled^(-1)(x) =  scaled(x) * scale  + x_min
    
    x = 1.0
    f(x_min) = 0.0
    """
end

function Base.show(io::IO, p::PolyFit{T}) where T<:Number
    print(io, "PolyFit{$T} Fit degree: $(p.deg) Fitted Coefficients: ",
              "$(p.coeffs) Number of fits: $(p.nfit) ")
    
    if p.scaled
        print(io, "Scaled: true, Scales: $(p.scales)\n")
    else
        print(io, "Scaled: false")
    end
end


function fit_poly(x::Vector{T}, y::VecOrMat{T}, deg::Int64,
                  scaled::Bool=true, dim::Int64=1) where T<:Number
    
    # TODO: make it general for y::Array{T,N}; transpose when dim == 1
    dim <= 0 && error("dim should be > 0!")
    
    n = ndims(y)
    
    @inbounds begin
    
    if n == 1
        dim > 1 && error("")
        yy = y
    elseif n == 2
        if dim == 1
            yy = y
        elseif dim == 2
            yy = transpose(y)
        else
            error("dim >= 2")
        end
    else
        error("Max 2 dim.")
    end

    nfit = size(yy, 2)

    if scaled
        search = hcat(x, yy)
        
        minmax = extrema(search, dims=1)
        
        scales = vec([Scale(m) for m in minmax])
        
        # f(x) = (x - x_min) / (x_max - x_min)
        
        xs = scales[1]
        design = collect( ((xx - xs.min) / xs.scale)^p for xx in x, p in deg:-1:0)
        
        if n == 1
            ys = scales[2]
            @simd for jj in eachindex(yy)
                yy[jj] = (yy[jj] - ys.min) / ys.scale
            end
        else
            rows, cols = size(yy)
            for jj in 1:cols
                ys = scales[jj + 1]
                @simd for ii in 1:rows
                    iscale = 1.0 / ys.scale
                    yy[ii,jj] = (yy[ii,jj] - ys.min) * iscale
                end
            end
        end
    else
        yy, xs, ys = y, nothing, nothing
        design = collect(xx^p for xx in x, p in deg:-1:0)
    end
    
    # @inbounds
    end
    
    return PolyFit{T}(design \ yy, deg, nfit, scaled, scales)
end


function eval_poly(p::PolyFit{T}, x::T) where T<:Number
    nfit, deg, coeffs = p.nfit, p.deg, p.coeffs
    
    ret = Vector{T}(undef, nfit)
    
    @inbounds begin
    
    if p.scaled
        scales = p.scales
        xs = scales[1]
        
        xx = (x - xs.min) / xs.scale
        if deg == 1
            for ii in 1:nfit
                ys = scales[ii + 1]
                ret[ii] = (coeffs[1, ii] * xx + coeffs[2, ii]) * ys.scale + ys.min
            end
            return ret
        else
            for jj in 1:nfit
                ret[jj] = coeffs[1, jj] * xx
                
                for ii in 2:deg
                    ret[jj] += coeffs[ii, jj] * xx
                end

                ys = scales[jj + 1]
                ret[jj] = (ret[jj] + coeffs[deg + 1, jj]) * ys.scale + ys.min
            end
            return ret
        # if deg == 1
        end
    else
        if deg == 1
            for ii in 1:nfit
                ret[ii] = coeffs[1, ii] * x + coeffs[2, ii]
            end
            return ret
        else
            for jj in 1:nfit
                ret[jj] = coeffs[1, jj] * x
                
                for ii in 2:deg
                    ret[jj] += coeffs[ii, jj] * x
                end
            end
            return ret
        # if deg == 1
        end
    # if p.scaled
    end
    
    # @inbounds
    end
# eval_poly
end


function eval_poly(p::PolyFit{T}, x::Vector{T}) where T<:Number
    nfit, nx, deg, coeffs = p.nfit, length(x), p.deg, p.coeffs
    
    ret = Matrix{T}(undef, nfit, nx)
    
    @inbounds begin
    
    if p.scaled
        scales = p.scales
        xs = scales[1]
        
        if deg == 1
            for jj in 1:nx
                xx = (x[jj] - xs.min) / xs.scale
                for ii in 1:nfit
                    ys = scales[ii + 1]
                    ret[ii, jj] = (coeffs[1, ii] * xx + coeffs[2, ii]) * ys.scale + ys.min
                end
            end
            return ret
        else
            for jj in 1:nx
                xx = (x[jj] - xs.min) / xs.scale
                
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, jj] * xx
                end
                
                for ii in 1:nfit
                    for kk in 2:deg
                        ret[ii, jj] += coeffs[kk, ii] * xx
                    end
                end
                
                ys = scales[jj + 1]
                
                for ii in 1:nfit
                    ret[ii, jj] = (ret[ii, jj] + coeffs[deg + 1, jj]) * ys.scale + ys.min
                end
            end
            return ret
        # if deg == 1
        end
    else
        if deg == 1
            for jj in 1:nx
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, ii] * x + coeffs[2, ii]
                end
            end
            return ret
        else
            for jj in 1:nx
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, jj] * x
                end
                
                for ii in 1:nfit
                    for kk in 2:deg
                        ret[ii, jj] += coeffs[kk, ii] * x
                    end
                end
            end
            return ret
        # if deg == 1
        end
    # if p.scaled
    end
    
    # @inbounds
    end
    
# eval_poly
end


function (p::PolyFit{T})(x::Union{T,Vector{T}}) where T<:Number
    return eval_poly(p, x)
end


function scale_back(p::Scale{T}, x::T) where T<:Number
    return x * p.scale + p.min
end


function scale_back(p::Scale{T}, x::Vector{T}) where T<:Number
    n, min, scale = length(x), p.min, p.scale
    xx::Vector{T}(undef, n)
    
    @inbounds @simd for ii = 1:n
        xx[ii] = x[ii] * scale + min
    end
    
    return xx
end

function scale_it(p::Scale{T}, x::T) where T<:Number
    return (x - p.min) / p.scale
end


function scale_it(p::Scale{T}, x::Vector{T}) where T<:Number
    n, min, iscale = length(x), p.min, 1.0 / p.scale
    xx::Vector{T}(undef, n)
    
    @inbounds @simd for ii = 1:n
        xx[ii] = (x[ii] - min) / iscale
    end
    
    return xx
end
=#


# module
end
```

## SatOrbit.jl

```julia
__precompile__(true)

module SatOrbit

using DelimitedFiles
using PolynomFit:fit_poly, PolyFit
using Serialization:serialize, deserialize

export read_orbits, fit_orbit, load_fit, Cart, Ellip, dot_product, ell_cart


const R_earth = 6372000.0;

const WA = 6378137.0
const WB = 6356752.3142

# (WA * WA - WB* WB) / WA / WA
const E2 = 6.694380e-03

const deg2rad = 1.745329e-02
const rad2deg = 5.729578e+01



"""
    orb = read_orbits(path::AbstractString, preproc::AbstractString)
    
    Read the orbit data (t, x(t), y(t), z(t)) stored in annotation files
    generated with DORIS or GAMMA preprocessing.
    
    # Example
"""
function read_orbits(path::String, preproc::String)
    if !isfile(path)
        error("$path does not exist!")
    end
    
    lines = readlines(path)
    
    if preproc == "doris"
        data_num = [(ii, line) for (ii, line) in enumerate(lines)
                               if startswith(line, "NUMBER_OF_DATAPOINTS:")]
        
        if length(data_num) != 1
            error("More than one or none of the lines contain the number of 
                   datapoints.")
        end
        
        idx = data_num[1][1]
        data_num = parse(Int, split(data_num[1][2], ":")[2])
        
        return transpose(readdlm(IOBuffer(join(lines[idx + 1:idx + data_num], "\n"))))
    elseif preproc == "gamma"
        error("Gamma processor not yet implemented!")
    else
        error("Unrecognized preprocessor option $preproc.");
    end
end


function fit_orbit(path::String, preproc::String,
                   savepath::Union{String,Nothing}=nothing, deg=3::Int64,
                   scaled=true::Bool)
    
    orb = read_orbits(path, preproc)
    
    fit = fit_poly(orb[1,:], orb[2:end,:], deg, scaled, 2)
    
    if savepath != nothing
        open(savepath, "w") do f
            serialize(f, fit)
        end
    end
    
    return fit
end


function load_fit(fit_file::String)
    open(fit_file, "r") do f
        fit = deserialize(f)
    end
end


struct Cart{T<:Number}
    x::T
    y::T
    z::T
end

struct Ellip{T<:Number}
    lon::T
    lat::T
    h::T
end


@inline function norm(x, y, z)
    return sqrt(x * x + y * y + z * z);
end


"""
Calculate dot product between satellite velocity vector and
and vector between ground position and satellite position.
"""
@inline function dot_product(orb::PolyFit, X::Float64, Y::Float64, Z::Float64,
                             time::Float64)
    x, y, z, vx, vy, vz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    npoly, deg, scaled, coeffs = orb.deg + 1, orb.deg, orb.scaled, orb.coeffs

    @show size(coeffs)
    
    @inbounds begin
    
    if deg == 1
        if scaled
            scales = orb.scales
            
            tt = (time - scales[1].min) / scales[1].scale

            x = (coeffs[1, 1] * tt + coeffs[2, 1]) * scales[2].scale + scales[2].min
            y = (coeffs[1, 2] * tt + coeffs[2, 2]) * scales[3].scale + scales[3].min
            z = (coeffs[1, 3] * tt + coeffs[2, 3]) * scales[4].scale + scales[4].min

            vx = coeffs[1, 1] * scales[2].scale;
            vy = coeffs[1, 2] * scales[3].scale;
            vz = coeffs[1, 3] * scales[4].scale;
        else
            x = coeffs[1, 1] * time + coeffs[2, 1]
            y = coeffs[1, 2] * time + coeffs[2, 2]
            z = coeffs[1, 3] * time + coeffs[2, 3]

            vx = coeffs[1, 1]
            vy = coeffs[1, 2]
            vz = coeffs[1, 3]
        end

    # deg != 1
    else
    
        if scaled
            x, y, z = coeffs[1, 1] * tt, coeffs[1, 2] * tt, coeffs[1, 3] * tt
            vx, vy, vz = coeffs[2, 1], coeffs[2, 2], coeffs[2, 3]

            for ii in 2:deg
                x += coeffs[ii, 1] * tt
                y += coeffs[ii, 2] * tt
                z += coeffs[ii, 3] * tt
            end

            for ii in 3:npoly
                vx += ii * coeffs[ii, 1] * tt^(ii-1)
                vy += ii * coeffs[ii, 2] * tt^(ii-1)
                vz += ii * coeffs[ii, 3] * tt^(ii-1)
            end

            x = (x + coeffs[npoly, 1]) * scales[2].scale + scales[2].min
            y = (y + coeffs[npoly, 2]) * scales[3].scale + scales[3].min
            z = (z + coeffs[npoly, 3]) * scales[4].scale + scales[4].min
            
            vx, vy, vz = vx * scales[2].scale, vy * scales[3].scale, vz * scales[4].scale

        # not scaled
        else
            x, y, z = coeffs[1, 1] * time, coeffs[1, 2] * time, coeffs[1, 3] * time
            vx, vy, vz = coeffs[2, 1], coeffs[2, 2], coeffs[2, 3]

            for ii in 2:deg
                x += coeffs[ii, 1] * time
                y += coeffs[ii, 2] * time
                z += coeffs[ii, 3] * time
            end

            for ii in 3:npoly
                vx += ii * coeffs[ii, 1] * time^(ii-1)
                vy += ii * coeffs[ii, 2] * time^(ii-1)
                vz += ii * coeffs[ii, 3] * time^(ii-1)
            end

            x = x + coeffs[npoly, 1]
            y = y + coeffs[npoly, 2]
            z = z + coeffs[npoly, 3]
        
        # not scaled
        end
    
    # deg != 1
    end
    
    # @inbounds
    end

    # satellite coordinates - surface coordinates
    dx, dy, dz = x - X, y - Y, z - Z
    
    # product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vx, vy, vz))
    
    return (vx * dx  + vy * dy  + vz * dz) * inorm

# dot_product
end


function ell_cart(e::Ellip)
    slat, clat, clon, slon, h = sin(e.lat), cos(e.lat), cos(e.lon), sin(e.lon), e.h
    
    n = WA / sqrt(1.0 - E2 * slat * slat)

    x = (              n + h) * clat * clon
    y = (              n + h) * clat * slon
    z = ( (1.0 - E2) * n + h) * slat

    return Cart(x, y, z)

# ell_cart
end


function cart_ell(c::Cart)
    x, y, z = c.x, c.z, c.z
    
    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    
    so = sin(o)
    co = cos(o)
    
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) )
    
    so = sin(o)
    co = cos(o);
    
    n = WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB)
    lat = o
    
    o = atan(y/x)
    
    if x < 0.0
        o += pi
    end
    
    lon = o
    h = p / co - n
    
    return Ellip(lon, lat, h)

# cart_ell
end


#function azi_inc(fit_file::AbstractString, coords::Array{T, 2},
#                 is_lonlat=true::Bool, max_iter=1000::Int) where T<:Real
    
#    t_start, t_stop, t_mean, coeffs, mean_coords, is_centered, deg = \
#    load_fit(fit_file)
    
#    inarg = (Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}, Cuint, Cuint, Cuint, Cuint, Cuint)
    
#    ndata = size(coords, 1)
    
#    ret = Array{Float64, 2}(ndata, 2)
    
#    ccall((:azi_inc, "libinsar"), (Void,), inarg, t_start, t_stop, t_mean,
#                                  coeffs, coords, mean_coords, ret, ndata,
#                                  is_centered, deg, max_iter, is_lonlat)
#    ret
#end

end
```
