#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory>

template<class T>
using uarray = std::unique_ptr<T[]>;

typedef uint8_t mem_var;
typedef std::shared_ptr<mem_var> shared_mem;

shared_mem make_shared(size_t num)
{
    shared_mem ret(new mem_var[num], std::default_delete<mem_var[]>());
    return ret;
}

#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)


#define m_malloc(num) malloc(num)
#define m_new(type, num) (type *) m_malloc(sizeof(type) * num)
#define m_free(ptr) free((ptr))

#define str_equal(str1, str2) (strcmp((str1), (str2)) == 0)

/**************
 * for macros *
 **************/

#define m_for(ii, max) for(size_t (ii) = (max); (ii)--; )
#define m_forz(ii, max) for(size_t (ii) = 0; (ii) < max; ++(ii))
#define m_fors(ii, min, max, step) for(size_t (ii) = (min); (ii) < (max); (ii) += (step))
#define m_for1(ii, min, max) for(size_t (ii) = (min); (ii) < (max); ++(ii))


#define m_check_fail(condition) \
do {                            \
    if ((condition))            \
        goto fail;              \
} while(0)


#define m_check(condition, expression) \
do {                                   \
    if ((condition))                   \
        (expression);                  \
} while(0)


/*
#ifdef __cplusplus
#define extern_begin extern "C" {
#define extern_end }
#else
*/
#define extern_begin
#define extern_end
//#endif


typedef void (*dtor)(void *);

#define del(obj)                \
do{                             \
    if ((obj)) {                \
        (obj)->dtor_((obj));    \
        m_free((obj));        \
        (obj) = NULL;           \
    }                           \
} while(0)


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

#endif
