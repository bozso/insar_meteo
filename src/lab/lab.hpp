#ifndef LAB_HPP
#define LAB_HPP


#include <chrono>
#include <algorithm>
#include <string>
#include <complex>
#include <memory>

typedef FILE* fileptr;

typedef unsigned char memtype;
typedef memtype* memptr;

// make_unique from https://herbsutter.com/gotw/_102/

template<typename T, typename ...Args>
std::unique_ptr<T> make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}

template<typename T, typename ...Args>
std::unique_ptr<T[]> make_array(long size, Args&& ...args)
{
    return std::unique_ptr<T[]>( new T[size]( std::forward<Args>(args)... ) );
}




struct Memory {
    Memory(): memory(nullptr), _size(0) {};
    
    Memory(long size): _size(size)
    {
        this->memory = make_array<memtype>(size);
    }
    
    ~Memory() = default;
    
    void realloc(long size)
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


template <class T>
void endswap(T& objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(&objp);
    std::reverse(memp, memp + sizeof(T));
}


typedef std::complex<double> cpx128;
typedef std::complex<float> cpx64;

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


// TODO: separate DataFile into DataFile (C++) and DataFileMeta (C, Python interface)

struct DataFile {
    typedef long idx;

    enum ftype {
        Unknown = 0,
        Array = 1,
        Matrix = 2,
        Vector = 3,
        Records = 4
    };
    
    int filetype;
    int *dtypes;
    long ntypes, recsize, nio;
    char *iomode, *datapath;
    memptr _file;
    
    Memory mem;
    memptr buffer;
    idx* offsets;

    fileptr file()
    {
        return reinterpret_cast<fileptr>(this->_file);
    }

    static std::string const dt2str(int type) noexcept;
    static std::string const ft2str(int type) noexcept;
    
    static dtype str2dt(std::string const& type) noexcept;
    static ftype str2ft(std::string const& type) noexcept;


    template<class T1, class T2>
    static T1 convert(memptr in)
    {
        return static_cast<T1>(*reinterpret_cast<T2*>(in));
    }

    
    DataFile() = default;
    void open();
    
    
    void read_rec();

    template<class T>
    void write_rec(T* obj);
    

    template<long ii, class T>
    T get()
    {
        memptr in = this->buffer + this->offsets[ii];
        
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
    
bool is_big_endian() noexcept;
long dtype_size(long type) noexcept;
void dtor_memory(Memory* mem);
void dtor_datafile(DataFile* datafile);

}


// guard
#endif
