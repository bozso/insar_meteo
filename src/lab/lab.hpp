#ifndef LAB_HPP
#define LAB_HPP


#include <string>
#include <complex>
#include <fstream>
#include <memory>
#include <functional>

using cpx128 = std::complex<double>;
using cpx64 = std::complex<float>;

enum dtype {
    Unknown = 0,
    Bool = 1,
    Int = 2,
    Long = 3,
    Size_t = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    UInt8 = 9,
    UInt16 = 10,
    UInt32 = 11,
    UInt64 = 12,
    Float32 = 13,
    Float64 = 14,
    Complex64 = 15,
    Complex128 = 16
};

/*
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
*/

typedef char memtype;

template<class T1, class T2>
static T1 convert(memtype* in)
{
    return static_cast<T1>(*reinterpret_cast<T2*>(in));
}


struct DataFile {
    typedef long idx;
    
    enum ftype {
        Unknown = 0,
        Array = 1,
        Matrix = 2,
        Vector = 3,
        Records = 4
    };
    
    ftype filetype;
    
    std::ios_base::openmode iomode;
    std::string datapath;
    std::unique_ptr<memtype[]> buffer;
    std::unique_ptr<idx[]> offsets;
    std::unique_ptr<dtype[]> dtypes;
    std::fstream file;          
    long ntypes, recsize, nio;

    
    static std::string const dt2str(int type) noexcept;
    static std::string const ft2str(int type) noexcept;
    
    static dtype str2dt(std::string const& type) noexcept;
    static ftype str2ft(std::string const& type) noexcept;
    
    DataFile(std::string const& datapath, long const recsize,
             long const ntypes, std::ios_base::openmode iomode);

    void readrec();
    

    template<long ii, class T>
    T get()
    {
        memtype* in = this->buffer.get() + this->offsets[ii];
        
        switch(this->dtypes[ii])
        {
            case Bool:
                return convert<T, bool>(in);
            case Int:
                return convert<T, int>(in);
            case Long:
                return convert<T, long>(in);
            case Size_t:
                return convert<T, size_t>(in);

            case Int8:
                return convert<T, int8_t>(in);
            case Int16:
                return convert<T, int16_t>(in);
            case Int32:
                return convert<T, int32_t>(in);
            case Int64:
                return convert<T, int64_t>(in);

            case UInt8:
                return convert<T, uint8_t>(in);
            case UInt16:
                return convert<T, uint16_t>(in);
            case UInt32:
                return convert<T, uint32_t>(in);
            case UInt64:
                return convert<T, uint64_t>(in);

            case Float32:
                return convert<T, float>(in);
            case Float64:
                return convert<T, float>(in);

            case Complex64:
                return convert<T, cpx64>(in);
            case Complex128:
                return convert<T, cpx128>(in);
        }
    }
    
    
    void close();
    ~DataFile() = default;
};


long dtype_size(dtype type) noexcept;


// guard
#endif
