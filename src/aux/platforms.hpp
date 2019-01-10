#ifndef PLATFORMS_H
#define PLATFORMS_H

/* 
 * Taken from julia/include/julia
 */

/*******************************************************************************
*                               Compiler                                       *
*******************************************************************************/

/*
 * Notes:
 *
 *  1. Checking for Intel's compiler should be done before checking for
 * Microsoft's. On Windows Intel's compiler also defines _MSC_VER as the
 * acknoledgement of the fact that it is integrated with Visual Studio.
 *
 *  2. Checking for MinGW should be done before checking for GCC as MinGW
 * pretends to be GCC.
 */
#if defined(__clang__)
#define _COMPILER_CLANG_
// Clang can also be used as a MinGW compiler
#if defined(__MINGW32__)
#define _COMPILER_MINGW_
#endif
#elif defined(__INTEL_COMPILER) || defined(__ICC)
#define _COMPILER_INTEL_
#elif defined(__MINGW32__)
#define _COMPILER_MINGW_
#elif defined(_MSC_VER)
#define _COMPILER_MICROSOFT_
#elif defined(__GNUC__)
#define _COMPILER_GCC_
#endif

/*******************************************************************************
*                               OS                                             *
*******************************************************************************/

#if defined(__FreeBSD__)
#define _OS_FREEBSD_
#elif defined(__linux__)
#define _OS_LINUX_
#elif defined(_WIN32) || defined(_WIN64)
#define _OS_WINDOWS_
#elif defined(__APPLE__) && defined(__MACH__)
#define _OS_DARWIN_
#endif

/*******************************************************************************
*                               Architecture                                   *
*******************************************************************************/

#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#define _CPU_X86_64_
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(_M_IX86) || defined(_X86_)
#define _CPU_X86_
#elif defined(__aarch64__)
#define _CPU_AARCH64_
#elif defined(__arm__) || defined(_M_ARM)
#define _CPU_ARM_
#elif defined(__PPC64__)
#define _CPU_PPC64_
#elif defined(_ARCH_PPC)
#define _CPU_PPC_
#endif


#if defined(_CPU_X86_64_)
#  define _P64
#elif defined(_CPU_X86_)
#  define _P32
#elif defined(_OS_WINDOWS_)
/* Not sure how to determine pointer size on Windows running ARM. */
#  if _WIN64
#    define _P64
#  else
#    define _P32
#  endif
#elif __SIZEOF_POINTER__ == 8
#    define _P64
#elif __SIZEOF_POINTER__ == 4
#    define _P32
#else
#  error pointer size not known for your platform / compiler
#endif


#endif
