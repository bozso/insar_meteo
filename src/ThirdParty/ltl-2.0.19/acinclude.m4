dnl CHECK FOR WORKING MMAP CALL
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_WORKING_MMAP],
[AC_CACHE_CHECK(whether mmap() works as expected,
ac_cv_working_mmap,
[AC_LANG_SAVE
 AC_LANG_C
 AC_TRY_RUN([
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
int main( void ) {
   int   fd;char* ptr;
   fd = open ("/etc/passwd", O_RDONLY, 0);
   if( ! ((void*)ptr = mmap (NULL, 512, PROT_READ,MAP_PRIVATE|MAP_FILE, fd, 0)) )
     exit(-1);
   return 0;}
],
 ac_cv_working_mmap=yes, ac_cv_working_mmap=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_working_mmap" = yes; then
  AC_DEFINE(HAVE_MMAP,,[define if mmap works])
fi
])


dnl CHECK FOR NAMESPACE SUPPORT
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_NAMESPACES],
[AC_CACHE_CHECK(whether the compiler supports namespaces,
ac_cv_cxx_namespaces,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([namespace Outer { namespace Inner { int i = 0; }}],
                [using namespace Outer::Inner; return i;],
 ac_cv_cxx_namespaces=yes, ac_cv_cxx_namespaces=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_namespaces" = yes; then
  AC_DEFINE(HAVE_NAMESPACES,,[define if the compiler supports namespaces])
fi
])


dnl CHECK FOR IEEE MATH
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_HAVE_IEEE_MATH],
[AC_CACHE_CHECK(whether the compiler supports the IEEE math library,
ac_cv_cxx_have_ieee_math,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_TRY_LINK([
#ifndef _ALL_SOURCE
 #define _ALL_SOURCE
#endif
#ifndef _XOPEN_SOURCE
 #define _XOPEN_SOURCE
#endif
#ifndef _XOPEN_SOURCE_EXTENDED
 #define _XOPEN_SOURCE_EXTENDED 1
#endif
#include <math.h>],[double x = 1.0; double y = 1.0;
acosh(x); asinh(x); atanh(x); expm1(x); erf(x); erfc(x); isnan(x);
j0(x); j1(x); lgamma(x); logb(x); log1p(x); rint(x); y0(x); y1(x);
return 0;],
 ac_cv_cxx_have_ieee_math=yes, ac_cv_cxx_have_ieee_math=no)
 LIBS="$ac_save_LIBS"
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_ieee_math" = yes; then
  AC_DEFINE(HAVE_IEEE_MATH,,[define if the compiler supports IEEE math library])
fi
])


dnl CHECK FOR STL
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_HAVE_STL],
[AC_CACHE_CHECK(whether the compiler has the Standard Template Library,
ac_cv_cxx_have_stl,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <list>
#include <deque>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[list<int> x; x.push_back(5);
list<int>::iterator iter = x.begin(); if (iter != x.end()) ++iter; return 0;],
 ac_cv_cxx_have_stl=yes, ac_cv_cxx_have_stl=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_stl" = no; then
  AC_MSG_ERROR(No Standard Template Library Found)
fi
])


dnl CHECK FOR numeric_Limits<>
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_HAVE_NUMERIC_LIMITS],
[AC_CACHE_CHECK(whether the compiler has numeric_limits<T>,
ac_cv_cxx_have_numeric_limits,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <limits>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[double e = numeric_limits<double>::epsilon(); return 0;],
 ac_cv_cxx_have_numeric_limits=yes, ac_cv_cxx_have_numeric_limits=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_numeric_limits" = yes; then
  AC_DEFINE(HAVE_NUMERIC_LIMITS,,[define if the compiler has numeric_limits<T>])
fi
])


dnl CHECK FOR PROPOSED 'RESTRICT' KEYWORD 
dnl ---------------------------------------------------------------------
dnl Try the official restrict keyword, then gcc's __restrict__, then
dnl SGI's __restrict.  __restrict has slightly different semantics than
dnl restrict (it's a bit stronger, in that __restrict pointers can't
dnl overlap even with non __restrict pointers), but I think it should be
dnl okay under the circumstances where restrict is normally used.

dnl define restrict_ to be whatever is supported

AC_DEFUN([AC_CXX_NCEG_RESTRICT],
[AC_CACHE_CHECK([for NCEG/C99 restrict keyword], acx_cv_c_restrict,
[acx_cv_c_restrict=unsupported
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 for acx_kw in restrict __restrict__ __restrict; do
   AC_TRY_COMPILE([], [float * $acx_kw x;], [acx_cv_c_restrict=$acx_kw; break])
 done
 AC_LANG_RESTORE
])
acx_kw="$acx_cv_c_restrict"
if test "$acx_kw" = unsupported; then acx_kw=""; fi
   AC_DEFINE_UNQUOTED(restrict_, $acx_kw, [Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.  Do not define if restrict is supported directly.])
if test "$acx_cv_c_restrict" != unsupported; then  
   AC_DEFINE(HAVE_NCEG_RESTRICT,,
             [define if  the compiler supports the NCEG/C99 restrict keyword])
fi
])



dnl CHECK FOR ENUM COMPUTATIONS
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_ENUM_COMPUTATIONS],
[AC_CACHE_CHECK(whether the compiler supports enum computations,
ac_cv_cxx_enum_computations,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
struct A { enum { a = 5, b = 7, c = 2 }; };
struct B { enum { a = 1, b = 6, c = 9 }; };
template<class T1, class T2> struct Z
{ enum { a = (T1::a > T2::a) ? T1::a : T2::b,
         b = T1::b + T2::b,
         c = (T1::c * T2::c + T2::a + T1::a)
       };
};],[
return (((int)Z<A,B>::a == 5)
     && ((int)Z<A,B>::b == 13)
     && ((int)Z<A,B>::c == 24)) ? 0 : 1;],
 ac_cv_cxx_enum_computations=yes, ac_cv_cxx_enum_computations=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_enum_computations" = no; then
  AC_MSG_ERROR(Compiler does not support enum computations)
fi
])


dnl CHECK FOR TEMPLATES PARTIAL SPECIALIZATION
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_PARTIAL_SPECIALIZATION],
[AC_CACHE_CHECK(whether the compiler supports partial specialization,
ac_cv_cxx_partial_specialization,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
template<class T, int N> class A            { public : enum e { z = 0 }; };
template<int N>          class A<double, N> { public : enum e { z = 1 }; };
template<class T>        class A<T, 2>      { public : enum e { z = 2 }; };
],[return (A<int,3>::z == 0) && (A<double,3>::z == 1) && (A<float,2>::z == 2);],
 ac_cv_cxx_partial_specialization=yes, ac_cv_cxx_partial_specialization=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_partial_specialization" = no; then
  AC_MSG_ERROR(Compiler does not support partial specialization of templates)
fi
])


dnl CHECK FOR TEMPLATES FULL SPECIALIZATION
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_FULL_SPECIALIZATION_SYNTAX],
[AC_CACHE_CHECK(whether the compiler supports full specializations,
ac_cv_cxx_full_specialization_syntax,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
template<class T> class A        { public : int f () const { return 1; } };
template<>        class A<float> { public:  int f () const { return 0; } };],[
A<float> a; return a.f();],
 ac_cv_cxx_full_specialization_syntax=yes, ac_cv_cxx_full_specialization_syntax=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_full_specialization_syntax" = no; then
  AC_MSG_ERROR(Compiler does not support full specialization of templates)
fi
])


dnl CHECK FOR MEMBER TEMPLATES
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_MEMBER_TEMPLATES],
[AC_CACHE_CHECK(whether the compiler supports member templates,
ac_cv_cxx_member_templates,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
template<class T, int N> class A
{ public:
  template<int N2> A<T,N> operator=(const A<T,N2>& z) { return A<T,N>(); }
};],[A<double,4> x; A<double,7> y; x = y; return 0;],
 ac_cv_cxx_member_templates=yes, ac_cv_cxx_member_templates=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_member_templates" = no; then
  AC_MSG_ERROR(Compiler does not support member templates)
fi
])



dnl CHECK FOR MEMBER TEMPLATE DECLARATIONS OUTSIDE CLASSES
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_MEMBER_TEMPLATES_OUTSIDE_CLASS],
[AC_CACHE_CHECK(whether the compiler supports member templates outside the class declaration,
ac_cv_cxx_member_templates_outside_class,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
template<class T, int N> class A
{ public :
  template<int N2> A<T,N> operator=(const A<T,N2>& z);
};
template<class T, int N> template<int N2>
A<T,N> A<T,N>::operator=(const A<T,N2>& z){ return A<T,N>(); }],[
A<double,4> x; A<double,7> y; x = y; return 0;],
 ac_cv_cxx_member_templates_outside_class=yes, ac_cv_cxx_member_templates_outside_class=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_member_templates_outside_class" = no; then
  AC_MSG_ERROR(Compiler does not support template declarations outside classes)
fi
])


dnl CHECK FOR TYPENAME KEYWORD
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_TYPENAME],
[AC_CACHE_CHECK(whether the compiler recognizes typename,
ac_cv_cxx_typename,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([template<typename T>class X {public:X(){}};],
[X<float> z; return 0;],
 ac_cv_cxx_typename=yes, ac_cv_cxx_typename=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_typename" = yes; then
  AC_DEFINE(HAVE_TYPENAME,,[define if the compiler recognizes typename])
fi
])


dnl CHECK FOR TEMPLATE QUALIFIED RETURN TYPES
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_TEMPLATE_QUALIFIED_RETURN_TYPE],
[AC_CACHE_CHECK(whether the compiler supports template-qualified return types,
ac_cv_cxx_template_qualified_return_type,
[AC_REQUIRE([AC_CXX_TYPENAME])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
#ifndef HAVE_TYPENAME
 #define typename
#endif
template<class X, class Y> struct promote_trait             { typedef X T; };
template<>                 struct promote_trait<int, float> { typedef float T; };
template<class T> class A { public : A () {} };
template<class X, class Y>
A<typename promote_trait<X,Y>::T> operator+ (const A<X>&, const A<Y>&)
{ return A<typename promote_trait<X,Y>::T>(); }
],[A<int> x; A<float> y; A<float> z = x + y; return 0;],
 ac_cv_cxx_template_qualified_return_type=yes, ac_cv_cxx_template_qualified_return_type=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_template_qualified_return_type" = no; then
  AC_MSG_ERROR(Compiler does not support template qualified return types)
fi
])


dnl CHECK FOR TEMPLATE PARTIAL ORDERING
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_PARTIAL_ORDERING],
[AC_CACHE_CHECK(whether the compiler supports partial ordering,
ac_cv_cxx_partial_ordering,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([
template<int N> struct I {};
template<class T> struct A
{  int r;
   template<class T1, class T2> int operator() (T1, T2)       { r = 0; return r; }
   template<int N1, int N2>     int operator() (I<N1>, I<N2>) { r = 1; return r; }
};],[A<float> x, y; I<0> a; I<1> b; return x (a,b) + y (float(), double());],
 ac_cv_cxx_partial_ordering=yes, ac_cv_cxx_partial_ordering=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_partial_ordering" = no; then
  AC_MSG_ERROR(Compiler does not support template partial ordering)
fi
])


dnl CHECK FOR GCC-STYLE BYTE SWAP INTRINSICS
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_HAVE_BSWAP_BUILTINS],
[AC_CACHE_CHECK(whether the compiler has gcc-style byte-swap builtins,
ac_cv_bswap_builtins,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_LINK([],[
int b = __builtin_bswap32 (0xdeaddead);
long d = __builtin_bswap64 (0xdeaddeaddeaddead);
return 0;
],
ac_cv_bswap_builtins=yes, ac_cv_bswap_builtins=no)
AC_LANG_RESTORE
])
if test "$ac_cv_bswap_builtins" = yes; then
  AC_DEFINE(HAVE_BSWAP_BUILTINS,,[define if the compiler has gcc-style byte-swap builtins])
fi
])


dnl CHECK FOR GCC-STYLE ATOMIC INC/SUB_AND_FETCH BUILTINS 
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_HAVE_ATOMIC_BUILTINS],
[AC_CACHE_CHECK(whether the compiler has gcc-style atomic add/sub_and_fetch builtins,
ac_cv_atomic_builtins,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_LINK([],[
int c=0, counter=0;
c = __sync_add_and_fetch( &counter, 1 );
c = __sync_sub_and_fetch( &counter, 1 );
return c;
],
ac_cv_atomic_builtins=yes, ac_cv_atomic_builtins=no)
AC_LANG_RESTORE
])
if test "$ac_cv_atomic_builtins" = yes; then
  AC_DEFINE(HAVE_ATOMIC_BUILTINS,,[define if the compiler has gcc-style atomic add/sub_and_fetch builtins])
fi
])


dnl CHECK FOR GCC-STYLE PREFETCH BUILTINS
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_HAVE_GCC_PREFETCH_BUILTINS],
[AC_CACHE_CHECK(whether the compiler has gcc-style prefetch builtins,
ac_cv_prefetch_builtins,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_LINK([],[
int c=0;
__builtin_prefetch(&c,0,1);
__builtin_prefetch(&c,1,1);
return c;
],
ac_cv_prefetch_builtins=yes, ac_cv_prefetch_builtins=no)
AC_LANG_RESTORE
])
if test "$ac_cv_prefetch_builtins" = yes; then
  AC_DEFINE(HAVE_GCC_PREFETCH_BUILTINS,,[define if the compiler has gcc-style prefetch builtins])
fi
])


dnl CHECK FOR GCC-STYLE attribute (vector_size) TYPES
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_HAVE_GCC_ATTRIBUTE_VECTOR_SIZE],
[AC_CACHE_CHECK(whether the compiler supports gcc-style __attribute__((vector size)),
ac_cv_attribute_vectorsize,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([],[
int c=0;
typedef char  vqi __attribute__ ((vector_size (16)));
typedef float vsf __attribute__ ((vector_size (16)));
vqi a;
vsf b;
return c;
],
ac_cv_attribute_vectorsize=yes, ac_cv_attribute_vectorsize=no)
AC_LANG_RESTORE
])
if test "$ac_cv_attribute_vectorsize" = yes; then
  AC_DEFINE(HAVE_GCC_ATTRIBUTE_VECTOR_SIZE,,[define if the compiler supports __attribute__((vector size))])
fi
])


dnl @synopsis AC_NEED_STDINT_H [( HEADER-TO-GENERATE [, HEDERS-TO-CHECK])]
dnl
dnl the "ISO C9X: 7.18 Integer types <stdint.h>" section requires the
dnl existence of an include file <stdint.h> that defines a set of 
dnl typedefs, especially uint8_t,int32_t,uintptr_t.
dnl Many older installations will not provide this file, but some will
dnl have the very same definitions in <inttypes.h>. In other enviroments
dnl we can use the inet-types in <sys/types.h> which would define the
dnl typedefs int8_t and u_int8_t respectivly.
dnl
dnl This macros will create a local "stdint.h" if it cannot find the
dnl global <stdint.h> (or it will create the headerfile given as an argument).
dnl In many cases that file will just have a singular "#include <inttypes.h>"
dnl statement, while in other environments it will provide the set of basic
dnl stdint's defined: 
dnl int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,intptr_t,uintptr_t
dnl int_least32_t.. int_fast32_t.. intmax_t
dnl which may or may not rely on the definitions of other files,
dnl or using the AC_COMPILE_CHECK_SIZEOF macro to determine the actual
dnl sizeof each type.
dnl
dnl if your header files require the stdint-types you will want to create an
dnl installable file package-stdint.h that all your other installable header
dnl may include. So if you have a library package named "mylib", just use
dnl      AC_NEED_STDINT(zziplib-stdint.h) 
dnl in configure.in and go to install that very header file in Makefile.am
dnl along with the other headers (mylib.h) - and the mylib-specific headers
dnl can simply use "#include <mylib-stdint.h>" to obtain the stdint-types.
dnl
dnl Remember, if the system already had a valid <stdint.h>, the generated
dnl file will include it directly. No need for fuzzy HAVE_STDINT_H things...
dnl
dnl @version $Id: acinclude.m4 550 2015-02-02 21:48:49Z drory $
dnl @author  Guido Draheim <guidod@gmx.de>       STATUS: used on new platforms
dnl Modified by Robby Dermody <robbyd@avalonent.org> REASON: Fixed warning about
dnl __AC_STDINT_H with newer versions of autoconf
dnl
AC_DEFUN([AC_NEED_STDINT_H],
[AC_MSG_CHECKING([for stdint-types])
 ac_cv_header_stdint="no-file"
 ac_cv_header_stdint_u="no-file"
 for i in $1 inttypes.h sys/inttypes.h sys/int_types.h stdint.h ; do
   AC_CHECK_TYPEDEF_(uint32_t, $i, [ac_cv_header_stdint=$i])
 done
 for i in $1 sys/types.h inttypes.h sys/inttypes.h sys/int_types.h ; do
   AC_CHECK_TYPEDEF_(u_int32_t, $i, [ac_cv_header_stdint_u=$i])
 done
 dnl debugging: __AC_MSG( !$ac_cv_header_stdint!$ac_cv_header_stdint_u! ...)

 ac_stdint_h=`echo ifelse($1, , stdint.h, $1)`
 if test "$ac_cv_header_stdint" != "no-file" ; then
   if test "$ac_cv_header_stdint" != "$ac_stdint_h" ; then
     AC_MSG_RESULT(found in $ac_cv_header_stdint)
     echo "#include <$ac_cv_header_stdint>" >$ac_stdint_h
     AC_MSG_RESULT(creating $ac_stdint_h - (just to include  $ac_cv_header_stdint) )
   else
     AC_MSG_RESULT(found in $ac_stdint_h)
   fi
   ac_cv_header_stdint_generated=false
 elif test "$ac_cv_header_stdint_u" != "no-file" ; then
   AC_MSG_RESULT(found u_types in $ac_cv_header_stdint_u)
   if test $ac_cv_header_stdint = "$ac_stdint_h" ; then
     AC_MSG_RESULT(creating $ac_stdint_h - includes $ac_cv_header_stdint, expect problems!)
   else
     AC_MSG_RESULT(creating $ac_stdint_h - (include inet-types in $ac_cv_header_stdint_u and re-typedef))
   fi
   cat >$ac_stdint_h <<EOF
#ifndef __NEED_STDINT_H
#define __NEED_STDINT_H 1
#include <stddef.h>
#include <$ac_cv_header_stdint_u>
/* int8_t int16_t int32_t defined by inet code */
typedef u_int8_t uint8_t;
typedef u_int16_t uint16_t;
typedef u_int32_t uint32_t;
/* it's a networkable system, but without any stdint.h */
/* hence it's an older 32-bit system... (a wild guess that seems to work) */
typedef u_int32_t uintptr_t;
typedef   int32_t  intptr_t;
EOF
   ac_cv_header_stdint_generated=true
 else
   AC_MSG_RESULT(not found, need to guess the types now... )
   AC_COMPILE_CHECK_SIZEOF(long, 32)
   AC_COMPILE_CHECK_SIZEOF(void*, 32)
   AC_MSG_RESULT( creating $ac_stdint_h - using detected values for sizeof long and sizeof void* )
   cat >$ac_stdint_h <<EOF

#ifndef __NEED_STDINT_H
#define __NEED_STDINT_H 1
/* ISO C 9X: 7.18 Integer types <stdint.h> */

#define __int8_t_defined  
typedef   signed char    int8_t;
typedef unsigned char   uint8_t;
typedef   signed short  int16_t;
typedef unsigned short uint16_t;
EOF

   if test "$ac_cv_sizeof_long" = "64" ; then
     cat >>$ac_stdint_h <<EOF

typedef   signed int    int32_t;
typedef unsigned int   uint32_t;
typedef   signed long   int64_t;
typedef unsigned long  uint64_t;
#define  int64_t  int64_t
#define uint64_t uint64_t
EOF

   else
    cat >>$ac_stdint_h <<EOF

typedef   signed long   int32_t;
typedef unsigned long  uint32_t;
EOF

   fi
   if test "$ac_cv_sizeof_long" != "$ac_cv_sizeof_voidp" ; then
     cat >>$ac_stdint_h <<EOF

typedef   signed int   intptr_t;
typedef unsigned int  uintptr_t;
EOF
   else
     cat >>$ac_stdint_h <<EOF

typedef   signed long   intptr_t;
typedef unsigned long  uintptr_t;
EOF
     ac_cv_header_stdint_generated=true
   fi
 fi   

 if "$ac_cv_header_stdint_generated" ; then
     cat >>$ac_stdint_h <<EOF

typedef  int8_t    int_least8_t;
typedef  int16_t   int_least16_t;
typedef  int32_t   int_least32_t;

typedef uint8_t   uint_least8_t;
typedef uint16_t  uint_least16_t;
typedef uint32_t  uint_least32_t;

typedef  int8_t    int_fast8_t;	
typedef  int32_t   int_fast16_t;
typedef  int32_t   int_fast32_t;

typedef uint8_t   uint_fast8_t;	
typedef uint32_t  uint_fast16_t;
typedef uint32_t  uint_fast32_t;

typedef long int       intmax_t;
typedef unsigned long uintmax_t;
#endif
EOF
  fi dnl
])


AC_DEFUN([AC_COMPILE_CHECK_SIZEOF],
[changequote(<<, >>)dnl
dnl The name to #define.
define(<<AC_TYPE_NAME>>, translit(sizeof_$1, [a-z *], [A-Z_P]))dnl
dnl The cache variable name.
define(<<AC_CV_NAME>>, translit(ac_cv_sizeof_$1, [ *], [_p]))dnl
changequote([, ])dnl
AC_MSG_CHECKING(size of $1)
AC_CACHE_VAL(AC_CV_NAME,
[for ac_size in 4 8 1 2 16 $2 ; do # List sizes in rough order of prevalence.
  AC_TRY_COMPILE([#include "confdefs.h"
#include <sys/types.h>
$2
], [switch (0) case 0: case (sizeof ($1) == $ac_size):;], AC_CV_NAME=$ac_size)
  if test x$AC_CV_NAME != x ; then break; fi
done
])
if test x$AC_CV_NAME = x ; then
  AC_MSG_ERROR([cannot determine a size for $1])
fi
AC_MSG_RESULT($AC_CV_NAME)
AC_DEFINE_UNQUOTED(AC_TYPE_NAME, $AC_CV_NAME, [The number of bytes in type $1])
undefine([AC_TYPE_NAME])dnl
undefine([AC_CV_NAME])dnl
])


AC_DEFUN([AC_CHECK_TYPEDEF_],
[dnl
ac_lib_var=`echo $1['_']$2 | sed 'y%./+-%__p_%'`
AC_CACHE_VAL(ac_cv_lib_$ac_lib_var,
[ eval "ac_cv_type_$ac_lib_var='not-found'"
  ac_cv_check_typedef_header=`echo ifelse([$2], , stddef.h, $2)`
  AC_TRY_COMPILE( [#include <$ac_cv_check_typedef_header>], 
	[int x = sizeof($1); x = x;],
        eval "ac_cv_type_$ac_lib_var=yes" ,
        eval "ac_cv_type_$ac_lib_var=no" )
  if test `eval echo '$ac_cv_type_'$ac_lib_var` = "no" ; then 
     ifelse([$4], , :, $4)
  else 
     ifelse([$3], , :, $3) 
  fi
])])

dnl AC_CHECK_TYPEDEF(TYPEDEF, HEADER [, ACTION-IF-FOUND,
dnl    [, ACTION-IF-NOT-FOUND ]])
AC_DEFUN([AC_CHECK_TYPEDEF],
[dnl
 AC_MSG_CHECKING([for $1 in $2])
 AC_CHECK_TYPEDEF_($1,$2,AC_MSG_RESULT(yes),AC_MSG_RESULT(no))dnl
])


dnl CHECK FOR std::complex<>
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_CXX_HAVE_COMPLEX],
[AC_CACHE_CHECK(whether the compiler has complex<T>,
ac_cv_cxx_have_complex,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[complex<float> a; complex<double> b; return 0;],
 ac_cv_cxx_have_complex=yes, ac_cv_cxx_have_complex=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_complex" = yes; then
  AC_DEFINE(HAVE_COMPLEX,,[define if the compiler has complex<T>])
fi
])

dnl CHECK FOR COMPILER OPTIONS
dnl ---------------------------------------------------------------------
# SYNOPSIS
#
#   AX_CHECK_COMPILER_FLAGS(FLAGS, [ACTION-SUCCESS], [ACTION-FAILURE])
#
# DESCRIPTION
#
#   Check whether the given compiler FLAGS work with the current language's
#   compiler, or whether they give an error. (Warnings, however, are
#   ignored.)
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
#
AC_DEFUN([AX_CHECK_COMPILER_FLAGS],
[AC_PREREQ(2.59) dnl for _AC_LANG_PREFIX
AC_MSG_CHECKING([whether _AC_LANG compiler accepts $1])
dnl Some hackery here since AC_CACHE_VAL can't handle a non-literal varname:
AS_LITERAL_IF([$1],
  [AC_CACHE_VAL(AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1]), [
      ax_save_FLAGS=$[]_AC_LANG_PREFIX[]FLAGS
      _AC_LANG_PREFIX[]FLAGS="$1"
      AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
        AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1])=yes,
        AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1])=no)
      _AC_LANG_PREFIX[]FLAGS=$ax_save_FLAGS])],
  [ax_save_FLAGS=$[]_AC_LANG_PREFIX[]FLAGS
   _AC_LANG_PREFIX[]FLAGS="$1"
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
     eval AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1])=yes,
     eval AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1])=no)
   _AC_LANG_PREFIX[]FLAGS=$ax_save_FLAGS])
eval ax_check_compiler_flags=$AS_TR_SH(ax_cv_[]_AC_LANG_ABBREV[]_flags_[$1])
AC_MSG_RESULT($ax_check_compiler_flags)
if test "x$ax_check_compiler_flags" = xyes; then
	m4_default([$2], :)
else
	m4_default([$3], :)
fi
])dnl AX_CHECK_COMPILER_FLAGS



dnl @synopsis AC_SYS_LARGEFILE_SENSITIVE
dnl
dnl checker whether the current system is sensitive to -Ddefines
dnl making off_t having different types/sizes. Automatically define
dnl a config.h symbol LARGEFILE_SENSITIVE if that is the case,
dnl otherwise leave everything as is. 
dnl
dnl This macro builds on top of AC_SYS_LARGEFILE to detect whether
dnl special options are neede to make the code use 64bit off_t - in
dnl many setups this will also make the code use 64bit off_t immediatly.
dnl
dnl The common use of a LARGEFILE_SENSITIVE config.h-define is to rename
dnl exported functions, usually adding a 64 to the original function name.
dnl Such renamings are only needed on systems being both (a) 32bit off_t
dnl by default and (b) implementing large.file extensions (as for unix98).
dnl
dnl a renaming section could look like this:
dnl  #if defined LARGEFILE_SENSITIVE && _FILE_OFFSET_BITS+0 == 64
dnl  #define zzip_open zzip_open64
dnl  #define zzip_seek zzip_seek64
dnl  #endif
dnl
dnl for libraries, it is best to take advantage of the prefix-config.h
dnl macro, otherwise you want to export a renamed LARGEFILE_SENSITIVE
dnl in an installed header file. -> see AX_PREFIX_CONFIG_H
dnl
dnl @, System Headers
dnl @Author Guido Draheim <guidod@gmx.de>
dnl @Version $Id: acinclude.m4 550 2015-02-02 21:48:49Z drory $

AU_ALIAS([AC_SYS_LARGEFILE_SENSITIVE], [AX_SYS_LARGEFILE_SENSITIVE])
AC_DEFUN([AX_SYS_LARGEFILE_SENSITIVE],[dnl
AC_REQUIRE([AC_SYS_LARGEFILE])dnl
# we know about some internals of ac_sys_largefile here...
AC_MSG_CHECKING(whether system differentiates 64bit off_t by defines)
ac_cv_sys_largefile_sensitive="no"
if test ".${ac_cv_sys_file_offset_bits-no}${ac_cv_sys_large_files-no}" != ".nono"
then ac_cv_sys_largefile_sensitive="yes"
  AC_DEFINE(LARGEFILE_SENSITIVE, 1,
  [whether the system defaults to 32bit off_t but can do 64bit when requested])
fi
AC_MSG_RESULT([$ac_cv_sys_largefile_sensitive])
])

dnl @synopsis AX_GCC_ARCHFLAG([PORTABLE?], [ACTION-SUCCESS], [ACTION-FAILURE])
dnl @summary find target architecture name for gcc -march/-mtune flags
dnl @category Misc
dnl
dnl This macro tries to guess the "native" arch corresponding to
dnl the target architecture for use with gcc's -march=arch or -mtune=arch
dnl flags.  If found, the cache variable $ax_cv_gcc_archflag is set to this
dnl flag and ACTION-SUCCESS is executed; otherwise $ax_cv_gcc_archflag is
dnl is set to "unknown" and ACTION-FAILURE is executed.  The default
dnl ACTION-SUCCESS is to add $ax_cv_gcc_archflag to the end of $CFLAGS.
dnl
dnl PORTABLE? should be either [yes] (default) or [no].  In the former case,
dnl the flag is set to -mtune (or equivalent) so that the architecture
dnl is only used for tuning, but the instruction set used is still
dnl portable.  In the latter case, the flag is set to -march (or equivalent)
dnl so that architecture-specific instructions are enabled.
dnl
dnl The user can specify --with-gcc-arch=<arch> in order to override
dnl the macro's choice of architecture, or --without-gcc-arch to
dnl disable this.
dnl
dnl When cross-compiling, or if $CC is not gcc, then ACTION-FAILURE is
dnl called unless the user specified --with-gcc-arch manually.
dnl
dnl Requires macros: AX_CHECK_COMPILER_FLAGS, AX_GCC_X86_CPUID
dnl
dnl The main emphasis here is on recent CPUs, on the principle that
dnl  doing high-performance computing on old hardware is uncommon.
dnl
dnl @version 2008-10-29
dnl @license GPLWithACException
dnl @author Steven G. Johnson <stevenj@alum.mit.edu> and Matteo Frigo.
AC_DEFUN([AX_GCC_ARCHFLAG],
[AC_REQUIRE([AC_PROG_CC])
AC_REQUIRE([AC_CANONICAL_HOST])

AC_ARG_WITH(gcc-arch, [AC_HELP_STRING([--with-gcc-arch=<arch>], [use architecture <arch> for gcc -march/-mtune, instead of guessing])], 
	ax_gcc_arch=$withval, ax_gcc_arch=yes)

AC_MSG_CHECKING([for gcc architecture flag])
AC_MSG_RESULT([])
AC_CACHE_VAL(ax_cv_gcc_archflag,
[
ax_cv_gcc_archflag="unknown"

if test "$GCC" = yes; then

if test "x$ax_gcc_arch" = xyes; then
ax_gcc_arch=""
if test "$cross_compiling" = no; then
case $host_cpu in
  i[[3456]]86*|x86_64*|amd64*) # use cpuid codes, in part from x86info-1.21 by D. Jones
     AX_GCC_X86_CPUID(0)
     AX_GCC_X86_CPUID(1)
     case $ax_cv_gcc_x86_cpuid_0 in
       *:756e6547:*:*) # Intel
          case $ax_cv_gcc_x86_cpuid_1 in
	    *5[[48]]?:*:*:*) ax_gcc_arch="pentium-mmx pentium" ;;
	    *5??:*:*:*) ax_gcc_arch=pentium ;;
	    *0?6[[3456]]?:*:*:*) ax_gcc_arch="pentium2 pentiumpro" ;;
	    *0?6a?:*[[01]]:*:*) ax_gcc_arch="pentium2 pentiumpro" ;;
	    *0?6a?:*[[234]]:*:*) ax_gcc_arch="pentium3 pentiumpro" ;;
	    *0?6[[9de]]?:*:*:*) ax_gcc_arch="pentium-m pentium3 pentiumpro" ;;
	    *0?6[[78b]]?:*:*:*) ax_gcc_arch="pentium3 pentiumpro" ;;
	    *0?6f?:*:*:*|*1?66?:*:*:*) ax_gcc_arch="core2 pentium-m pentium3 pentiumpro" ;;
	    *1?6[[7d]]?:*:*:*) ax_gcc_arch="penryn core2 pentium-m pentium3 pentiumpro" ;;
	    *1?6[[aef]]?:*:*:*|*2?6[[5cef]]?:*:*:*) ax_gcc_arch="corei7 core2 pentium-m pentium3 pentiumpro" ;;
	    *1?6c?:*:*:*|*[[23]]?66?:*:*:*) ax_gcc_arch="atom core2 pentium-m pentium3 pentiumpro" ;;
	    *2?6[[ad]]?:*:*:*) ax_gcc_arch="corei7-avx corei7 core2 pentium-m pentium3 pentiumpro" ;;
	    *0?6??:*:*:*) ax_gcc_arch=pentiumpro ;;
	    *6??:*:*:*) ax_gcc_arch="core2 pentiumpro" ;;
	    ?000?f3[[347]]:*:*:*|?000?f4[1347]:*:*:*|?000?f6?:*:*:*)
		case $host_cpu in
	          x86_64*) ax_gcc_arch="nocona pentium4 pentiumpro" ;;
	          *) ax_gcc_arch="prescott pentium4 pentiumpro" ;;
	        esac ;;
	    ?000?f??:*:*:*) ax_gcc_arch="pentium4 pentiumpro";;
          esac ;;
       *:68747541:*:*) # AMD
          case $ax_cv_gcc_x86_cpuid_1 in
	    *5[[67]]?:*:*:*) ax_gcc_arch=k6 ;;
	    *5[[8d]]?:*:*:*) ax_gcc_arch="k6-2 k6" ;;
	    *5[[9]]?:*:*:*) ax_gcc_arch="k6-3 k6" ;;
	    *60?:*:*:*) ax_gcc_arch=k7 ;;
	    *6[[12]]?:*:*:*) ax_gcc_arch="athlon k7" ;;
	    *6[[34]]?:*:*:*) ax_gcc_arch="athlon-tbird k7" ;;
	    *67?:*:*:*) ax_gcc_arch="athlon-4 athlon k7" ;;
	    *6[[68a]]?:*:*:*)
	       AX_GCC_X86_CPUID(0x80000006) # L2 cache size
	       case $ax_cv_gcc_x86_cpuid_0x80000006 in
                 *:*:*[[1-9a-f]]??????:*) # (L2 = ecx >> 16) >= 256
			ax_gcc_arch="athlon-xp athlon-4 athlon k7" ;;
                 *) ax_gcc_arch="athlon-4 athlon k7" ;;
	       esac ;;
	    ?00??f[[4cef8b]]?:*:*:*) ax_gcc_arch="athlon64 k8" ;;
	    ?00??f5?:*:*:*) ax_gcc_arch="opteron k8" ;;
	    ?00??f7?:*:*:*) ax_gcc_arch="athlon-fx opteron k8" ;;
	    ?00??f??:*:*:*) ax_gcc_arch="k8" ;;
	    ?05??f??:*:*:*) ax_gcc_arch="btver1 amdfam10 k8" ;;
	    ?06??f??:*:*:*) ax_gcc_arch="bdver1 amdfam10 k8" ;;
	    *f??:*:*:*) ax_gcc_arch="amdfam10 k8" ;;
          esac ;;
	*:746e6543:*:*) # IDT
	   case $ax_cv_gcc_x86_cpuid_1 in
	     *54?:*:*:*) ax_gcc_arch=winchip-c6 ;;
	     *58?:*:*:*) ax_gcc_arch=winchip2 ;;
	     *6[[78]]?:*:*:*) ax_gcc_arch=c3 ;;
	     *69?:*:*:*) ax_gcc_arch="c3-2 c3" ;;
	   esac ;;
     esac
     if test x"$ax_gcc_arch" = x; then # fallback
	case $host_cpu in
	  i586*) ax_gcc_arch="native pentium" ;;
	  i686*) ax_gcc_arch="native pentiumpro" ;;
          x86_64*|amd64*) ax_gcc_arch="native" ;;
        esac
     fi 
     ;;

  sparc*)
     AC_PATH_PROG([PRTDIAG], [prtdiag], [prtdiag], [$PATH:/usr/platform/`uname -i`/sbin/:/usr/platform/`uname -m`/sbin/])
     cputype=`(((grep cpu /proc/cpuinfo | cut -d: -f2) ; ($PRTDIAG -v |grep -i sparc) ; grep -i cpu /var/run/dmesg.boot ) | head -n 1) 2> /dev/null`
     cputype=`echo "$cputype" | tr -d ' -' |tr $as_cr_LETTERS $as_cr_letters`
     case $cputype in
         *ultrasparciv*) ax_gcc_arch="ultrasparc4 ultrasparc3 ultrasparc v9" ;;
         *ultrasparciii*) ax_gcc_arch="ultrasparc3 ultrasparc v9" ;;
         *ultrasparc*) ax_gcc_arch="ultrasparc v9" ;;
         *supersparc*|*tms390z5[[05]]*) ax_gcc_arch="supersparc v8" ;;
         *hypersparc*|*rt62[[056]]*) ax_gcc_arch="hypersparc v8" ;;
         *cypress*) ax_gcc_arch=cypress ;;
     esac ;;

  alphaev5) ax_gcc_arch=ev5 ;;
  alphaev56) ax_gcc_arch=ev56 ;;
  alphapca56) ax_gcc_arch="pca56 ev56" ;;
  alphapca57) ax_gcc_arch="pca57 pca56 ev56" ;;
  alphaev6) ax_gcc_arch=ev6 ;;
  alphaev67) ax_gcc_arch=ev67 ;;
  alphaev68) ax_gcc_arch="ev68 ev67" ;;
  alphaev69) ax_gcc_arch="ev69 ev68 ev67" ;;
  alphaev7) ax_gcc_arch="ev7 ev69 ev68 ev67" ;;
  alphaev79) ax_gcc_arch="ev79 ev7 ev69 ev68 ev67" ;;

  powerpc*)
     cputype=`((grep cpu /proc/cpuinfo | head -n 1 | cut -d: -f2 | cut -d, -f1 | sed 's/ //g') ; /usr/bin/machine ; /bin/machine; grep CPU /var/run/dmesg.boot | head -n 1 | cut -d" " -f2) 2> /dev/null`
     cputype=`echo $cputype | sed -e 's/ppc//g;s/ *//g'`
     case $cputype in
       *750*) ax_gcc_arch="750 G3" ;;
       *740[[0-9]]*) ax_gcc_arch="$cputype 7400 G4" ;;
       *74[[4-5]][[0-9]]*) ax_gcc_arch="$cputype 7450 G4" ;;
       *74[[0-9]][[0-9]]*) ax_gcc_arch="$cputype G4" ;;
       *970*) ax_gcc_arch="970 G5 power4";;
       *POWER4*|*power4*|*gq*) ax_gcc_arch="power4 970";;
       *POWER5*|*power5*|*gr*|*gs*) ax_gcc_arch="power5 power4 970";;
       603ev|8240) ax_gcc_arch="$cputype 603e 603";;
       *Cell*) ax_gcc_arch="cellppu cell";;
       *) ax_gcc_arch="$cputype native" ;;
     esac
     ax_gcc_arch="$ax_gcc_arch powerpc"
     ;;
esac
fi # not cross-compiling
fi # guess arch

if test "x$ax_gcc_arch" != x -a "x$ax_gcc_arch" != xno; then
for arch in $ax_gcc_arch; do
  if test "x[]m4_default([$1],yes)" = xyes; then # if we require portable code
    flags="-mtune=$arch"
    # -mcpu=$arch and m$arch generate nonportable code on every arch except
    # x86.  And some other arches (e.g. Alpha) don't accept -mtune.  Grrr.
    case $host_cpu in i*86|x86_64*|amd64*) flags="$flags -mcpu=$arch -m$arch";; esac
  else
    flags="-march=$arch -mcpu=$arch -m$arch"
  fi
  for flag in $flags; do
    AX_CHECK_COMPILER_FLAGS($flag, [ax_cv_gcc_archflag=$flag; break])
  done
  test "x$ax_cv_gcc_archflag" = xunknown || break
done
fi

fi # $GCC=yes
])
AC_MSG_CHECKING([for gcc architecture flag])
AC_MSG_RESULT($ax_cv_gcc_archflag)
if test "x$ax_cv_gcc_archflag" = xunknown; then
  m4_default([$3],:)
else
  m4_default([$2], [CFLAGS="$CFLAGS $ax_cv_gcc_archflag"])
fi
])

# ===========================================================================
#     http://www.gnu.org/software/autoconf-archive/ax_gcc_x86_cpuid.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_GCC_X86_CPUID(OP)
#
# DESCRIPTION
#
#   On Pentium and later x86 processors, with gcc or a compiler that has a
#   compatible syntax for inline assembly instructions, run a small program
#   that executes the cpuid instruction with input OP. This can be used to
#   detect the CPU type.
#
#   On output, the values of the eax, ebx, ecx, and edx registers are stored
#   as hexadecimal strings as "eax:ebx:ecx:edx" in the cache variable
#   ax_cv_gcc_x86_cpuid_OP.
#
#   If the cpuid instruction fails (because you are running a
#   cross-compiler, or because you are not using gcc, or because you are on
#   a processor that doesn't have this instruction), ax_cv_gcc_x86_cpuid_OP
#   is set to the string "unknown".
#
#   This macro mainly exists to be used in AX_GCC_ARCHFLAG.
AC_DEFUN([AX_GCC_X86_CPUID],
[AC_REQUIRE([AC_PROG_CC])
AC_LANG_PUSH([C])
AC_CACHE_CHECK(for x86 cpuid $1 output, ax_cv_gcc_x86_cpuid_$1,
 [AC_RUN_IFELSE([AC_LANG_PROGRAM([#include <stdio.h>], [
     int op = $1, eax, ebx, ecx, edx;
     FILE *f;
      __asm__("cpuid"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "a" (op));
     f = fopen("conftest_cpuid", "w"); if (!f) return 1;
     fprintf(f, "%x:%x:%x:%x\n", eax, ebx, ecx, edx);
     fclose(f);
     return 0;
])],
     [ax_cv_gcc_x86_cpuid_$1=`cat conftest_cpuid`; rm -f conftest_cpuid],
     [ax_cv_gcc_x86_cpuid_$1=unknown; rm -f conftest_cpuid],
     [ax_cv_gcc_x86_cpuid_$1=unknown])])
AC_LANG_POP([C])
])

dnl CHECK FOR APPLE VecLib/Accelerate framework
dnl ---------------------------------------------------------------------

AC_DEFUN([AC_HAVE_VECTOR_SUPPORT],
[AC_CACHE_CHECK(whether the compiler supports vector types,
ac_cv_vector_support,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([],[
typedef float     vsf __attribute__ ((vector_size (16)));
vsf a = {0.f,1.f,2.f,3.f};
return 0;
],
ac_cv_vector_support=yes, ac_cv_vector_support=no)
AC_LANG_RESTORE
])
if test "$ac_cv_vector_support" = yes; then
  AC_DEFINE(HAVE_VECTOR_SUPPORT,,[define if the compiler has gnu-style vector support])
fi
])

AC_DEFUN([AC_HAVE_APPLE_VECLIB],
[AC_CACHE_CHECK(whether the system provides Apple's vecLib and Accelerate frameworks,
ac_cv_apple_veclib,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -framework Accelerate"
 AC_TRY_RUN([
#include "Accelerate/Accelerate.h"
int main( void ) {
typedef float     vsf __attribute__ ((vector_size (16)));
vsf a = {0.f,1.f,2.f,3.f};
vsf b = vsinf( a );
return 0;}
],
ac_cv_apple_veclib=yes, ac_cv_apple_veclib=no)
LIBS="$ac_save_LIBS"
AC_LANG_RESTORE
])
if test "$ac_cv_apple_veclib" = yes; then
  AC_DEFINE(HAVE_APPLE_VECLIB,,[define if system provides Apple's vecLib and Accelerate framework])
fi
])



dnl CHECK FOR BLAS & LAPACK (Intel, AMD, Apple, Sun, ATLAS, Alpha, ...)
dnl ---------------------------------------------------------------------

dnl @synopsis ACX_BLAS
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl @author Adam Piatyszek <ediap@users.sourceforge.net>
dnl @version 2007-02-15
dnl
dnl This macro looks for a library that implements the BLAS
dnl linear-algebra interface (see http://www.netlib.org/blas/). On
dnl success, it sets the BLAS_LIBS output variable to hold the
dnl requisite library linkages. Besides, it defines HAVE_BLAS.
dnl
dnl To link with BLAS, you should link with:
dnl
dnl 	$BLAS_LIBS $LIBS
dnl
dnl Many libraries are searched for, e.g. MKL, ACML or ATLAS. The
dnl user may also use --with-blas=<lib> in order to use some specific
dnl BLAS library <lib>. In order to link successfully, however, be
dnl aware that you will probably need to use the same Fortran compiler
dnl (which can be set via the F77 env. var.) as was used to compile the
dnl BLAS library.
dnl
dnl This macro requires autoconf 2.50 or later.

AC_DEFUN([ACX_BLAS], [
AC_PREREQ(2.50)

test "x$sgemm" = x && sgemm=sgemm_
test "x$dgemm" = x && dgemm=dgemm_

# Initialise local variables
acx_blas_ok=no
blas_mkl_ok=no
blas_acml_ok=no
blas_atlas_ok=no
acx_zdotu=auto

# Parse "--with-blas=<lib>" option
AC_ARG_WITH(blas,
  [AS_HELP_STRING([--with-blas@<:@=LIB@:>@],
                  [use BLAS library, optionally specified by LIB])])
case $with_blas in
  yes | "") ;;
  no) acx_blas_ok=disabled ;;
  -* | */* | *.a | *.so | *.so.* | *.o) BLAS_LIBS="$with_blas" ;;
  *) BLAS_LIBS="-l$with_blas" ;;
esac


# First, check BLAS_LIBS environment variable
if test "$acx_blas_ok" = no; then
  if test "x$BLAS_LIBS" != x; then
    save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
    AC_MSG_CHECKING([for $sgemm in $BLAS_LIBS])
    AC_TRY_LINK_FUNC($sgemm, [acx_blas_ok=yes])
    AC_MSG_RESULT($acx_blas_ok)
    # Try to use MY_FLIBS
    if test "$acx_blas_ok" = no; then
      LIBS="$LIBS$MY_FLIBS"
      AC_MSG_CHECKING([for $sgemm in $BLAS_LIBS$MY_FLIBS])
      AC_TRY_LINK_FUNC($sgemm,
        [acx_blas_ok=yes; BLAS_LIBS="$BLAS_LIBS$MY_FLIBS"],
        [BLAS_LIBS=""])
      AC_MSG_RESULT($acx_blas_ok)
    fi
    LIBS="$save_LIBS"
  fi
fi

## BLAS linked to by default?  (happens on some supercomputers)
#if test $acx_blas_ok = no; then
#  save_LIBS="$LIBS"; LIBS="$LIBS"
#  AC_CHECK_FUNC($sgemm, [acx_blas_ok=yes])
#  LIBS="$save_LIBS"
#fi

# BLAS in MKL library?
# (http://www.intel.com/cd/software/products/asmo-na/eng/perflib/mkl/index.htm)
if test "$acx_blas_ok" = no; then
  AC_CHECK_LIB([mkl], [$sgemm],
    [acx_blas_ok=yes; blas_mkl_ok=yes; acx_zdotu=void;
       BLAS_LIBS="-lmkl -lguide -lpthread"],
    [],
    [-lguide -lpthread])
fi

# BLAS in ACML library? (http://developer.amd.com/acml.aspx)
# +------------+-----------+---------------------+
# |            |  32-bit   |       64-bit        |
# +------------+-----------+---------------------+
# | GCC <  4.2 | -lacml    | -lacml -lacml_mv    |
# | GCC >= 4.2 | -lacml_mp | -lacml_mp -lacml_mv |
# +------------+-----------+---------------------+
if test "$acx_blas_ok" = no; then
  save_LIBS="$LIBS"; LIBS="$LIBS$MY_FLIBS"
  AC_CHECK_LIB([acml], [$sgemm],
    [acx_blas_ok=yes; blas_acml_ok=yes; BLAS_LIBS="-lacml$MY_FLIBS"],
    [AC_CHECK_LIB([acml_mp], [$sgemm],
       [acx_blas_ok=yes; blas_acml_ok=yes; BLAS_LIBS="-lacml_mp$MY_FLIBS"],
       [AC_CHECK_LIB([acml], [$dgemm],
          [acx_blas_ok=yes; blas_acml_ok=yes;
             BLAS_LIBS="-lacml -lacml_mv$MY_FLIBS"],
          [AC_CHECK_LIB([acml_mp], [$dgemm],
             [acx_blas_ok=yes; blas_acml_ok=yes;
                BLAS_LIBS="-lacml_mp -lacml_mv$MY_FLIBS"],
             [], [-lacml_mv])],
          [], [-lacml_mv])],
       [])],
    [])
  LIBS="$save_LIBS"
fi

# BLAS in Apple vecLib library?
if test "$acx_blas_ok" = no; then
        save_LIBS="$LIBS"; LIBS="-framework Accelerate $LIBS"
	echo "checking for $sgemm in framework vecLib"
        AC_CHECK_FUNC($sgemm, [acx_blas_ok=yes;BLAS_LIBS="-framework Accelerate"])        
        LIBS="$save_LIBS"
fi

# BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
if test "$acx_blas_ok" = no; then
  save_LIBS="$LIBS"; LIBS="$LIBS$MY_FLIBS"
  AC_CHECK_LIB(atlas, ATL_xerbla,
    [AC_CHECK_LIB(f77blas, $sgemm,
      [AC_CHECK_LIB(cblas, cblas_dgemm,
        [acx_blas_ok=yes; blas_atlas_ok=yes;
          BLAS_LIBS="-lcblas -lf77blas -latlas$MY_FLIBS"],
        [], [-lf77blas -latlas])],
      [], [-latlas])])
  LIBS="$save_LIBS"
fi

# # BLAS in Alpha CXML library?
# if test $acx_blas_ok = no; then
#   AC_CHECK_LIB(cxml, $sgemm, [acx_blas_ok=yes; BLAS_LIBS="-lcxml"])
# fi

# # BLAS in Alpha DXML library? (now called CXML, see above)
# if test $acx_blas_ok = no; then
#   AC_CHECK_LIB(dxml, $sgemm, [acx_blas_ok=yes; BLAS_LIBS="-ldxml"])
# fi

# BLAS in Sun Performance library?
#if test $acx_blas_ok = no; then
#  if test "x$GCC" != xyes; then # only works with Sun CC
#    AC_CHECK_LIB(sunmath, acosp,
#      [AC_CHECK_LIB(sunperf, $sgemm,
#        [acx_blas_ok=yes; BLAS_LIBS="-xlic_lib=sunperf -lsunmath"], [],
#          [-lsunmath])])
#  fi
#fi

# # BLAS in SCSL library?  (SGI/Cray Scientific Library)
# if test $acx_blas_ok = no; then
#   AC_CHECK_LIB(scs, $sgemm, [acx_blas_ok=yes; BLAS_LIBS="-lscs"])
# fi

# # BLAS in SGIMATH library?
# if test $acx_blas_ok = no; then
#   AC_CHECK_LIB(complib.sgimath, $sgemm,
#     [acx_blas_ok=yes; BLAS_LIBS="-lcomplib.sgimath"])
# fi

# # BLAS in IBM ESSL library? (requires generic BLAS lib, too)
# if test $acx_blas_ok = no; then
#   AC_CHECK_LIB(blas, $sgemm,
#     [AC_CHECK_LIB(essl, $sgemm,
#       [acx_blas_ok=yes; BLAS_LIBS="-lessl -lblas"], [], [-lblas $FLIBS])])
# fi

# Generic BLAS library?
if test "$acx_blas_ok" = no; then
  AC_CHECK_LIB(blas, $sgemm, [acx_blas_ok=yes; BLAS_LIBS="-lblas"],
    [AC_CHECK_LIB(blas, $dgemm, [acx_blas_ok=yes; BLAS_LIBS="-lblas$MY_FLIBS"],
      [], [$MY_FLIBS])])
fi

## if BLAS is found check what kind of BLAS it is
#if test "$acx_blas_ok" = yes && test "$blas_mkl_ok" = no \
#    && test "$blas_acml_ok" = no && test "$blas_atlas_ok" = no; then
#  save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
#  AC_MSG_CHECKING([for MKLGetVersion in $BLAS_LIBS])
#  AC_TRY_LINK_FUNC(MKLGetVersion, [blas_mkl_ok=yes; acx_zdotu=void])
#  AC_MSG_RESULT($blas_mkl_ok)
#  if test "$blas_mkl_ok" = no; then
#    AC_MSG_CHECKING([for acmlversion in $BLAS_LIBS])
#    AC_TRY_LINK_FUNC(acmlversion, [blas_acml_ok=yes])
#    AC_MSG_RESULT($blas_acml_ok)
#  fi
#  if test "$blas_mkl_ok" = no && test "$blas_acml_ok" = no; then
#    AC_MSG_CHECKING([for ATL_xerbla in $BLAS_LIBS])
#    AC_TRY_LINK_FUNC(ATL_xerbla, [blas_atlas_ok=yes])
#    AC_MSG_RESULT($blas_atlas_ok)
#  fi
#  LIBS="$save_LIBS"
#fi

AC_SUBST(BLAS_LIBS)

# Finally, define HAVE_BLAS
if test "$acx_blas_ok" = yes; then
  AC_DEFINE(HAVE_BLAS, 1, [Define if you have a BLAS library.])
  if test "$blas_mkl_ok" = yes; then
    AC_DEFINE(HAVE_BLAS_MKL, 1, [Define if you have an MKL BLAS library.])
  fi
  if test "$blas_acml_ok" = yes; then
    AC_DEFINE(HAVE_BLAS_ACML, 1, [Define if you have an ACML BLAS library.])
  fi
  if test "$blas_atlas_ok" = yes; then
    AC_DEFINE(HAVE_BLAS_ATLAS, 1, [Define if you have an ATLAS BLAS library.])
  fi
else
  if test "$acx_blas_ok" != disabled; then
    AC_MSG_WARN([cannot find any BLAS library, which is required by LAPACK.
You can override this warning by using "--without-blas" option.])
  fi
fi

])dnl ACX_BLAS


dnl @synopsis ACX_LAPACK
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl @author Adam Piatyszek <ediap@users.sourceforge.net>
dnl @version 2007-02-15
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/). On
dnl success, it sets the LAPACK_LIBS output variable to hold the
dnl requisite library linkages. Besides, it defines HAVE_LAPACK
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACK_LIBS $BLAS_LIBS $LIBS
dnl
dnl in that order. BLAS_LIBS is the output variable of the ACX_BLAS
dnl macro, called automatically.
dnl
dnl The user may also use --with-lapack=<lib> in order to use some
dnl specific LAPACK library <lib>. In order to link successfully,
dnl however, be aware that you will probably need to use the same
dnl Fortran compiler (which can be set via the F77 env. var.) as was
dnl used to compile the LAPACK and BLAS libraries.

AC_DEFUN([ACX_LAPACK], [
AC_REQUIRE([ACX_BLAS])

test "x$cheev" = x && cheev=cheev_

# Initialise local variables
# We cannot use LAPACK if BLAS is not found
if test "$acx_blas_ok" != yes; then
  acx_lapack_ok=noblas
else
  acx_lapack_ok=no
fi

# Parse "--with-lapack=<lib>" option
AC_ARG_WITH(lapack,
  [AS_HELP_STRING([--with-lapack@<:@=LIB@:>@], [use LAPACK library, optionally specified by LIB])])
case $with_lapack in
  yes | "") ;;
  no) acx_lapack_ok=disabled ;;
  -* | */* | *.a | *.so | *.so.* | *.o) LAPACK_LIBS="$with_lapack" ;;
  *) LAPACK_LIBS="-l$with_lapack" ;;
esac

# First, check LAPACK_LIBS environment variable
if test "x$LAPACK_LIBS" != x; then
  save_LIBS="$LIBS"; LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS"
  AC_MSG_CHECKING([for $cheev in $LAPACK_LIBS])
  AC_TRY_LINK_FUNC($cheev, [acx_lapack_ok=yes], [LAPACK_LIBS=""])
  AC_MSG_RESULT($acx_lapack_ok)
  LIBS="$save_LIBS"
fi

# LAPACK linked to by default?  (it is sometimes included in BLAS)
if test "$acx_lapack_ok" = no; then
  save_LIBS="$LIBS"; LIBS="$LIBS $BLAS_LIBS"
  AC_CHECK_FUNC($cheev, [acx_lapack_ok=yes])
  LIBS="$save_LIBS"
fi

# LAPACK in MKL library?
# (http://www.intel.com/cd/software/products/asmo-na/eng/perflib/mkl/index.htm)
if test "$acx_lapack_ok" = no; then
  save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
  AC_CHECK_LIB(mkl_lapack32, $cheev,
    [acx_lapack_ok=yes; LAPACK_LIBS="-lmkl_lapack32 -lmkl_lapack64"],
    [AC_CHECK_LIB(mkl_lapack, $cheev,
      [acx_lapack_ok=yes; LAPACK_LIBS="-lmkl_lapack"])],
    [-lmkl_lapack64])
  LIBS="$save_LIBS"
fi

# Generic LAPACK library?
for lapack in lapack lapack_rs6k; do
  if test "$acx_lapack_ok" = no; then
    save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
    AC_CHECK_LIB($lapack, $cheev, [acx_lapack_ok=yes; LAPACK_LIBS="-l$lapack"])
    LIBS="$save_LIBS"
  fi
done

AC_SUBST(LAPACK_LIBS)

# Finally, define HAVE_LAPACK
if test "$acx_lapack_ok" = yes; then
  AC_DEFINE(HAVE_LAPACK, 1, [Define if you have LAPACK library.])
fi

])dnl ACX_LAPACK
