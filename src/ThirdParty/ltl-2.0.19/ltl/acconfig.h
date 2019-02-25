/* ltl/acconfig.h.  Generated from acconfig.h.in by configure.  */
/* ltl/acconfig.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* define if system provides Apple's vecLib and Accelerate framework */
#define HAVE_APPLE_VECLIB /**/

/* define if the compiler has gcc-style atomic add/sub_and_fetch builtins */
#define HAVE_ATOMIC_BUILTINS /**/

/* Define if you have a BLAS library. */
#define HAVE_BLAS 1

/* Define if you have an ACML BLAS library. */
/* #undef HAVE_BLAS_ACML */

/* Define if you have an ATLAS BLAS library. */
/* #undef HAVE_BLAS_ATLAS */

/* Define if you have an MKL BLAS library. */
/* #undef HAVE_BLAS_MKL */

/* define if the compiler has gcc-style byte-swap builtins */
#define HAVE_BSWAP_BUILTINS /**/

/* define if the compiler has complex<T> */
#define HAVE_COMPLEX /**/

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <float.h> header file. */
#define HAVE_FLOAT_H 1

/* define if the compiler supports __attribute__((vector size)) */
#define HAVE_GCC_ATTRIBUTE_VECTOR_SIZE /**/

/* define if the compiler has gcc-style prefetch builtins */
#define HAVE_GCC_PREFETCH_BUILTINS /**/

/* Define to 1 if you have the `getrusage' function. */
#define HAVE_GETRUSAGE 1

/* define if the compiler supports IEEE math library */
#define HAVE_IEEE_MATH /**/

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <ios> header file. */
#define HAVE_IOS 1

/* Define if you have LAPACK library. */
#define HAVE_LAPACK 1

/* Define to 1 if you have the <limits> header file. */
#define HAVE_LIMITS 1

/* Define to 1 if the type `long double' works and has more range or precision
   than `double'. */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if the type `long double' works and has more range or precision
   than `double'. */
#define HAVE_LONG_DOUBLE_WIDER 1

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `mktime' function. */
#define HAVE_MKTIME 1

/* Define to 1 if you have the `mmap' function. */
#define HAVE_MMAP 1

/* define if the compiler supports namespaces */
#define HAVE_NAMESPACES /**/

/* define if the compiler supports the NCEG/C99 restrict keyword */
#define HAVE_NCEG_RESTRICT /**/

/* define if the compiler has numeric_limits<T> */
#define HAVE_NUMERIC_LIMITS /**/

/* Define to 1 if you have the <pty.h> header file. */
/* #undef HAVE_PTY_H */

/* Define to 1 if you have the `snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define to 1 if you have the <sstream> header file. */
#define HAVE_SSTREAM 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strftime' function. */
#define HAVE_STRFTIME 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <strstream> header file. */
#define HAVE_STRSTREAM 1

/* Define to 1 if you have the <sys/mman.h> header file. */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* define if the compiler recognizes typename */
#define HAVE_TYPENAME /**/

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the <util.h> header file. */
#define HAVE_UTIL_H 1

/* define if the compiler has gnu-style vector support */
#define HAVE_VECTOR_SUPPORT /**/

/* whether the system defaults to 32bit off_t but can do 64bit when requested
   */
/* #undef LARGEFILE_SENSITIVE */

/* Multithreading support */
/* #undef LTL_MULTITHREAD */

/* Name of package */
#define PACKAGE "ltl"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "ltl"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "ltl 2.0.19"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "ltl"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.0.19"

/* The number of bytes in type long */
/* #undef SIZEOF_LONG */

/* The number of bytes in type void* */
/* #undef SIZEOF_VOIDP */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
/* #undef TM_IN_SYS_TIME */

/* Version number of package */
#define VERSION "2.0.19"

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Enable large inode numbers on Mac OS X 10.5.  */
#ifndef _DARWIN_USE_64_BIT_INODE
# define _DARWIN_USE_64_BIT_INODE 1
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
/* #undef _FILE_OFFSET_BITS */

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to equivalent of C99 restrict keyword, or to nothing if this is not
   supported. Do not define if restrict is supported directly. */
#define restrict_ __restrict__

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
