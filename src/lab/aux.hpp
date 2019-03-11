#ifndef AUX_HPP
#define AUX_HPP

#include <iostream>
#include <chrono>
#include <algorithm>
#include <memory>


// #define exc(type, 

namespace aux {

/********************
 * Endianness check *
 ********************/

bool is_big_endian();


template<class T>
std::unique_ptr<T[]> uarray(size_t size)
{
    return std::unique_ptr<T[]>(new T[size]);
}


template <class T>
void endswap(T& objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(&objp);
    std::reverse(memp, memp + sizeof(T));
}


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

#ifdef m_get_impl

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
    std::cout << "Elapsed time: " << this->elapsed() << " seconds.\n";
}


bool is_big_endian()
{
    short word = 0x4321;
    if ((*(char *)& word) != 0x21 )
    {
        return true;
    }
    else
    {
        return false;
    }
}

// implementation
#endif

// namespace
}



#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)

/**************
 * for macros *
 **************/

#define m_for(ii, max) for(size_t (ii) = (max); (ii)--; )
#define m_forz(ii, max) for(size_t (ii) = 0; (ii) < max; ++(ii))
#define m_fors(ii, min, max, step) for(size_t (ii) = (min); (ii) < (max); (ii) += (step))
#define m_for1(ii, min, max) for(size_t (ii) = (min); (ii) < (max); ++(ii))


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


// guard
#endif
