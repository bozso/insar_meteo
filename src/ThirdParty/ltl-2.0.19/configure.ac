dnl Process this file with autoconf to produce a configure script.
AC_INIT(ltl, 2.0.19)
AC_CONFIG_SRCDIR(ltl/marray.h)
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([tar-ustar])
AC_CONFIG_HEADER(ltl/acconfig.h)

dnl Checks for C++ compiler and deal with compiler options
AC_LANG_CPLUSPLUS
test ".$CXXFLAGS" = "." && CXXFLAGS=""
AC_PROG_CXX
CXXVECFLAGS=""
BASECXX=`basename $CXX`
CXXFLAGS_MARCH=""
AX_CHECK_COMPILER_FLAGS([-march=native],[CXXFLAGS_NATIVE="-march=native"],[CXXFLAGS_NATIVE=""])
AX_GCC_ARCHFLAG([no], [CXXFLAGS_ARCH="$ax_cv_gcc_archflag"], [CXXFLAGS_ARCH=""])
AX_CHECK_COMPILER_FLAGS([-msse3],[CXXVECFLAGS_386="-msse3"],[CXXVECFLAGS_386=""])
case "$BASECXX" in
  g++*)
	CXXFLAGS_MARCH=$CXXFLAGS_NATIVE
    if test "x$CXXFLAGS_MARCH" = "x"; then
    	echo "gcc option -march=native does not work, trying to find out cpuid ..."
		CXXFLAGS_MARCH=$CXXFLAGS_ARCH
    fi
	echo "architecture-specific gcc options: " $CXXFLAGS_MARCH
    CXXFLAGS_1="-O2 -Wall -fstrict-aliasing $CXXFLAGS_MARCH"
    case "$target" in
       *darwin*)
	       case "$target" in
	            *powerpc*)
	            	CXXFLAGS_1=$CXXFLAGS_1" -Wno-long-double -mdynamic-no-pic"
	                CXXVECFLAGS=-maltivec
	            	;;
	            *i?86*)
	    	        CXXVECFLAGS=$CXXVECFLAGS_386
	        	    ;;
	       esac
	       ;;
       *x86_64*)
           CXXVECFLAGS=$CXXVECFLAGS_386
       	   ;;
       *i?86*)
           CXXVECFLAGS=$CXXVECFLAGS_386
       	   ;;
    esac
    test ".$CXXFLAGS" = "." && CXXFLAGS=$CXXFLAGS_1
    ;;
	
  clang*)
  	CXXFLAGS_MARCH=$CXXFLAGS_NATIVE
	echo "architecture-specific clang++ options: " $CXXFLAGS_MARCH
    CXXFLAGS_1="-O3 -Wall $CXXFLAGS_MARCH"
  	case "$target" in
       *x86_64*)
           CXXVECFLAGS=$CXXVECFLAGS_386
       	   ;;
       *i?86*)
           CXXVECFLAGS=$CXXVECFLAGS_386
       	   ;;
    esac  	
    test ".$CXXFLAGS" = "." && CXXFLAGS=$CXXFLAGS_1
  	;;
  xlC*)
    test ".$CXXFLAGS" = "." && CXXFLAGS="-g -O3 -Q -qansialias -qstrict" 
    ;;
  CC)
    case "$target" in
       *solaris*)
       CXXEXTFLAGS=`fpversion | grep Use | cut -d \" -f 2`
       test ".$CXXFLAGS" = "." && CXXFLAGS="-g -fast -xalias_level=compatible $CXXEXTFLAGS"
     ;;
    esac
    ;;
  icc|icpc)
    test ".$CXXFLAGS" = "." && CXXFLAGS="-g -O2 -ansi_alias -restrict -ip"
    CXXVECFLAGS=-xO
    ;;
 esac

AC_ARG_ENABLE([multithread],[  --enable-multithread    compile with multithreading support (default no)],[ENABLE_MULTITHREAD=yes],[ENABLE_MULTITHREAD=no])

if test $ENABLE_MULTITHREAD = "yes"; then
   AC_DEFINE([LTL_MULTITHREAD],[],[Multithreading support])
fi

dnl check for large file support
AC_SYS_LARGEFILE_SENSITIVE
AC_SYS_LARGEFILE

AC_PROG_CXXCPP
AC_PROG_RANLIB

dnl Checks for libraries.

dnl compiler characteristics.
AC_LANG_CPLUSPLUS
AC_CXX_NAMESPACES
AC_C_LONG_DOUBLE
AC_CXX_NCEG_RESTRICT
AC_C_BIGENDIAN
AC_NEED_STDINT_H(ltl/misc/stdint_hack.h)

dnl Checks for header files.
dnl AC_LANG_C
AC_HEADER_STDC
AC_CHECK_HEADERS(math.h fcntl.h string.h unistd.h float.h sys/mman.h)
AC_CHECK_HEADERS(pty.h)
AC_CHECK_HEADERS(util.h)
AC_CHECK_FUNCS(mmap mktime strftime getrusage snprintf)
dnl AC_LANG_CPLUSPLUS

AC_CHECK_LIB(util, openpty, need_libutil=yes)
if test "$need_libutil" = "yes"; then
  UTIL_LIBS=-lutil
fi

AC_CXX_HAVE_STL
AC_CHECK_HEADERS(strstream sstream ios limits)
AC_CXX_HAVE_IEEE_MATH
AC_CXX_HAVE_NUMERIC_LIMITS
AC_CXX_HAVE_COMPLEX

dnl Checks for typedefs and structures.
AC_TYPE_SIZE_T
AC_STRUCT_TM

dnl check compiler for reasonalbe template support.
AC_CXX_ENUM_COMPUTATIONS
AC_CXX_FULL_SPECIALIZATION_SYNTAX
AC_CXX_PARTIAL_SPECIALIZATION
AC_CXX_MEMBER_TEMPLATES
AC_CXX_MEMBER_TEMPLATES_OUTSIDE_CLASS
AC_CXX_TEMPLATE_QUALIFIED_RETURN_TYPE
AC_CXX_PARTIAL_ORDERING

dnl check for additional libraries and features
echo
AC_HAVE_ATOMIC_BUILTINS
AC_HAVE_BSWAP_BUILTINS
AC_HAVE_GCC_PREFETCH_BUILTINS
AC_HAVE_GCC_ATTRIBUTE_VECTOR_SIZE
AC_HAVE_VECTOR_SUPPORT
AC_HAVE_APPLE_VECLIB

# Checks for BLAS (ACML, MKL, ATLAS or NetLib's reference)
ACX_BLAS
ACX_LAPACK

AM_CONDITIONAL(APPLE_VECLIB, test $ac_cv_apple_veclib = yes)
AM_CONDITIONAL(VECTOR_SUPPORT, test $ac_cv_vector_support = yes)
AM_CONDITIONAL(BLAS_SUPPORT, test $acx_blas_ok = yes)
AM_CONDITIONAL(LAPACK_SUPPORT, test $acx_lapack_ok = yes)
AM_CONDITIONAL(COMPLEX_SUPPORT, test $ac_cv_cxx_have_complex = yes)

# Set LIBS
test "x$BLAS_LIBS" != x && LIBS="$BLAS_LIBS $LIBS"
test "x$LAPACK_LIBS" != x && LIBS="$LAPACK_LIBS $LIBS"
test "x$UTIL_LIBS" != x && LIBS="$LIBS $UTIL_LIBS"

echo

AC_SUBST(CXXVECFLAGS)
AC_OUTPUT(Makefile ltl/Makefile doc/Makefile test/Makefile)

# Display configuration summary
AC_MSG_NOTICE([displaying configuration summary

------------------------------------------------------------------------------
$PACKAGE-$VERSION library configuration:
------------------------------------------------------------------------------

Directories:
  - prefix ......... : ${prefix}
  - exec_prefix .... : ${exec_prefix}
  - includedir ..... : ${includedir}
  - libdir ......... : ${libdir}
  - datarootdir .... : ${datarootdir}
  - docdir ......... : ${docdir}

Compiler/linker flags/libs/defs:
  - CXX ............ : ${CXX}
  - F77 ............ : ${F77}
  - CXXFLAGS ....... : ${CXXFLAGS}
  - CXXVECFLAGS .... : ${CXXVECFLAGS}
  - CPPFLAGS ....... : ${CPPFLAGS}
  - LDFLAGS ........ : ${LDFLAGS}
  - LIBS ........... : ${LIBS}

Features:
  - complex types .. : ${ac_cv_cxx_have_complex}
  - vectorization .. : ${ac_cv_vector_support}
  - attribute vsize  : ${ac_cv_attribute_vectorsize}
  - atomic builtins  : ${ac_cv_atomic_builtins}
  - bswap builtins . : ${ac_cv_bswap_builtins}
  - prefetch builtins: ${ac_cv_prefetch_builtins}

Options:
  - thread safety .. : ${ENABLE_MULTITHREAD}

External libs:
  - Apple Accelerate : ${ac_cv_apple_veclib}
  - BLAS ........... : ${acx_blas_ok}
    * MKL .......... : ${blas_mkl_ok}
    * ACML ......... : ${blas_acml_ok}
    * ATLAS ........ : ${blas_atlas_ok}
  - LAPACK ......... : ${acx_lapack_ok}

------------------------------------------------------------------------------
Now type 'make && make install' to build and install $PACKAGE-$VERSION library
------------------------------------------------------------------------------
])

echo
echo "Configure done."
echo
