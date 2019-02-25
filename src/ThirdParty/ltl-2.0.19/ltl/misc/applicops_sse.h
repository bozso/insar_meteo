/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: applicops_sse.h 556 2015-03-09 16:25:40Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C)  Niv Drory <drory@mpe.mpg.de>
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2, or (at your option)
* any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
*
* ---------------------------------------------------------------------
*
*/


#if !defined(__LTL_APPLICOPS_H__)
#error "<ltl/misc/applicops_sse.h> must be included after <ltl/misc/applicops.h> !"
#endif

#if !defined(__SSE2__)
#error "<ltl/misc/applicops_sse.h> needs SSE2 extensions to be enabled.\nConsult your compiler manual on how to do so\nUse the flag(s) -msse2 in GCC"
#endif

#ifndef __LTL_APPLICOPS_SIMD_H__
#define __LTL_APPLICOPS_SIMD_H__

// we assume at least SSE2 is available
#include <xmmintrin.h>   // SSE
#include <emmintrin.h>   // SSE2
//#ifdef __SSE3__
//#  include <pmmintrin.h>  // SSE3
//#endif
#ifdef __SSE4_1__
#  include <smmintrin.h>  // SSE4.1
#endif

#ifdef HAVE_APPLE_VECLIB
// Apple provides vector versions of standard lib math functions ;-)
#  include <Accelerate/Accelerate.h>
#endif

namespace ltl {

/*! \file applicops_sse.h

  Specializations of applicative templates using the MMX/SSE/SSE2
  vector instructions on x86 (Pentium4 and above)
*/

//@{

/*!
   traits mapping between scalar and vector types
   for vectorizing applicops.
*/
#if defined(HAVE_GCC_ATTRIBUTE_VECTOR_SIZE)
typedef char               vqi __attribute__ ((vector_size (16)));
typedef unsigned char      vqu __attribute__ ((vector_size (16)));
typedef short              vhi __attribute__ ((vector_size (16)));
typedef unsigned short     vhu __attribute__ ((vector_size (16)));
typedef int                vsi __attribute__ ((vector_size (16)));
typedef unsigned int       vsu __attribute__ ((vector_size (16)));
typedef long long          vdi __attribute__ ((vector_size (16)));
typedef unsigned long long vdu __attribute__ ((vector_size (16)));
typedef float              vsf __attribute__ ((vector_size (16)));
typedef double             vdf __attribute__ ((vector_size (16)));
#else
typedef __m128i vqi;
typedef __m128i vqu;
typedef __m128i vhi;
typedef __m128i vhu;
typedef __m128i vsi;
typedef __m128i vsu;
typedef __m128i vdi;
typedef __m128i vdu;
typedef __m128  vsf;
typedef __m128d vdf;
#endif


/*!
 * Type traits for vectors of given elemen type.
 *
 * These define the follwing types and methods:
 *
 * \code
 *  typedef T          value_type;          // The type of the vector's elements
 *  typedef Tint_type  int_value_type;      // The integer type with the same number of elems per vector
 *  typedef TTT        vec_type;            // The type denoting the vector itself
 *  typedef __m128i    arg_type;            // The type of the arguments to the SSE intrinsics
 *  static inline vec_type init(const T x); // Method returning a vector with all elems set to x
 * \endcode
 */
template<typename T> struct vec_trait
{
};

template<> struct vec_trait<bool>
{
      typedef bool       value_type;
      typedef value_type int_value_type;
      typedef vsi        vec_type;
      typedef __m128i    arg_type;
      static inline vec_type init(const bool x)
      {
         return (vec_type)_mm_set1_epi32(x);
      }
};

template<> struct vec_trait<char>
{
      typedef char       value_type;
      typedef value_type int_value_type;
      typedef vqi        vec_type;
      typedef __m128i    arg_type;
      static inline vec_type init(const char x)
      {
         return (vec_type)_mm_set1_epi8(x);
      }
};

template<> struct vec_trait<short>
{
      typedef short      value_type;
      typedef value_type int_value_type;
      typedef vhi        vec_type;
      typedef __m128i    arg_type;
      static inline vec_type init(const short x)
      {
         return (vec_type)_mm_set1_epi16(x);
      }
};

template<> struct vec_trait<int>
{
      typedef int        value_type;
      typedef value_type int_value_type;
      typedef vsi        vec_type;
      typedef __m128i    arg_type;
      static inline vec_type init(const int x)
      {
         return (vec_type)_mm_set1_epi32(x);
      }
};

template<> struct vec_trait<long long>
{
      typedef long long  value_type;
      typedef value_type int_value_type;
      typedef vdi        vec_type;
      typedef __m128i    arg_type;
      static inline vec_type init(const long long x)
      {
#ifdef __INTEL_COMPILER
         // the intel compiler stubbornly resists casting long long to __m64.
         union { long long tmp; __m64 v; } u;
         u.tmp = x;
         return _mm_set1_epi64(u.v);
#else
         return (vec_type)_mm_set1_epi64x(x);
#endif
      }
};

template<> struct vec_trait<unsigned char>
{
      typedef unsigned char value_type;
      typedef value_type    int_value_type;
      typedef vqu           vec_type;
      typedef __m128i       arg_type;
      static inline vec_type init(const unsigned char x)
      {
         return (vec_type)_mm_set1_epi8(x);
      }
};

template<> struct vec_trait<unsigned short>
{
      typedef unsigned short value_type;
      typedef value_type     int_value_type;
      typedef vhu            vec_type;
      typedef __m128i        arg_type;
      static inline vec_type init(const unsigned short x)
      {
         return (vec_type)_mm_set1_epi16(x);
      }
};

template<> struct vec_trait<unsigned int>
{
      typedef unsigned int value_type;
      typedef value_type   int_value_type;
      typedef vsu          vec_type;
      typedef __m128i      arg_type;
      static inline vec_type init(const unsigned int x)
      {
         return (vec_type)_mm_set1_epi32(x);
      }
};

template<> struct vec_trait<unsigned long long>
{
      typedef unsigned long long value_type;
      typedef value_type         int_value_type;
      typedef vdu                vec_type;
      typedef __m128i            arg_type;
      static inline vec_type init(const unsigned long long x)
      {
#ifdef __INTEL_COMPILER
         // the intel compiler stubbornly resists casting long long to __m64.
         union { long long tmp; __m64 v; } u;
         u.tmp = x;
         return _mm_set1_epi64(u.v);
#else
         return (vec_type)_mm_set1_epi64x(x);
#endif
      }
};

template<> struct vec_trait<float>
{
      typedef float  value_type;
      typedef int    int_value_type;
      typedef vsf    vec_type;
      typedef __m128 arg_type;
      static inline vec_type init(const float x)
      {
         return (vec_type)_mm_set1_ps(x);
      }
};

template<> struct vec_trait<double>
{
      typedef double    value_type;
      typedef long long int_value_type;
      typedef vdf       vec_type;
      typedef __m128d   arg_type;
      static inline vec_type init(const double x)
      {
         return (vec_type)_mm_set1_pd(x);
      }
};

//! vector type for given element type
#define VEC_TYPE(T)          vec_trait<T>::vec_type
#define T_VEC_TYPE(T)        typename VEC_TYPE(T)

//! integer type of same size for given type (for results of logical operations)
#define INT_TYPE(T)          vec_trait<T>::int_value_type
#define T_INT_TYPE(T)        typename INT_TYPE(T)

//! vector type of integers of same length as given element type
#define INT_VEC_TYPE(T)      VEC_TYPE(typename vec_trait<T>::int_value_type)
#define T_INT_VEC_TYPE(T)    typename INT_VEC_TYPE(T)

//! type of the arguments to the MMX intrinsics for this vector type
#define VARGT(T)             vec_trait<T>::arg_type
#define VARGTLL              VARGT(long long)
//! initialize a vector with all elements = val
#define VEC_INIT(T,val)      vec_trait<T>::init(val)
#define VEC_ZERO(T)          (vec_trait<T>::vec_type)_mm_setzero_si128()
#define T_VEC_ZERO(T)        (typename vec_trait<T>::vec_type)_mm_setzero_si128()

//! number of elements of a vector of given type
#define VEC_LEN(T)           ((int)(sizeof(VEC_TYPE(T))/sizeof(T)))
#define T_VEC_LEN(T)         ((int)(sizeof(T_VEC_TYPE(T))/sizeof(T)))

//! union of vector type and array of scalars
#define VEC_UNION(T)         union{ VEC_TYPE(T) v; T a[VEC_LEN(T)]; }
#define T_VEC_UNION(T)       union{ T_VEC_TYPE(T) v; T a[T_VEC_LEN(T)]; }
//@}


/*!
 * SSE has no vec_sel. Implement it in terms of AND/OR
 */
//@{
template<typename T>
static inline T_VEC_TYPE(T) sse_vec_sel( const T_VEC_TYPE(T) a, const T_VEC_TYPE(T) b, const T_INT_VEC_TYPE(T) mask )
{
   return (T_VEC_TYPE(T))_mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(mask,b) );
}
//@}

//! vec_not
//@{
static inline VEC_TYPE(float) sse_vec_not( const VEC_TYPE(float) a )
{
   return (VEC_TYPE(float))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
static inline VEC_TYPE(double) sse_vec_not( const VEC_TYPE(double) a )
{
   return (VEC_TYPE(double))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
static inline VEC_TYPE(int) sse_vec_not( const VEC_TYPE(int) a )
{
   return (VEC_TYPE(int))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
#ifndef __INTEL_COMPILER
static inline VEC_TYPE(long long) sse_vec_not( const VEC_TYPE(long long) a )
{
   return (VEC_TYPE(long long))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
static inline VEC_TYPE(short) sse_vec_not( const VEC_TYPE(short) a )
{
   return (VEC_TYPE(short))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
static inline VEC_TYPE(char) sse_vec_not( const VEC_TYPE(char) a )
{
   return (VEC_TYPE(char))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)VEC_INIT(unsigned int,0xFFFFFFFF) );
}
#endif
//@}

//! vec_andnot
//@{
static inline VEC_TYPE(float) sse_vec_andnot( const VEC_TYPE(float) a, const VEC_TYPE(float) b )
{
   return (VEC_TYPE(float))_mm_andnot_ps( a, b );
}
static inline VEC_TYPE(double) sse_vec_andnot( const VEC_TYPE(double) a, const VEC_TYPE(double) b )
{
   return (VEC_TYPE(double))_mm_andnot_pd( a, b );
}
static inline VEC_TYPE(int) sse_vec_andnot( const VEC_TYPE(int) a, const VEC_TYPE(int) b )
{
   return (VEC_TYPE(int))_mm_andnot_si128( (VARGTLL)a, (VARGTLL)b );
}
//@}

//! integer vec_cmpne
//@{
static inline VEC_TYPE(char) sse_vec_cmpneq_epi8( const VEC_TYPE(char) a, const VEC_TYPE(char) b )
{
   return (VEC_TYPE(char))sse_vec_not( (VARGTLL)_mm_cmpeq_epi8( (VARGTLL)a, (VARGTLL)b ) );
}
static inline VEC_TYPE(short) sse_vec_cmpneq_epi16( const VEC_TYPE(short) a, const VEC_TYPE(short) b )
{
   return (VEC_TYPE(short))sse_vec_not( (VARGTLL)_mm_cmpeq_epi16( (VARGTLL)a, (VARGTLL)b ) );
}
static inline VEC_TYPE(int) sse_vec_cmpneq_epi32( const VEC_TYPE(int) a, const VEC_TYPE(int) b )
{
   return (VEC_TYPE(int))sse_vec_not( (VARGTLL)_mm_cmpeq_epi32( (VARGTLL)a, (VARGTLL)b ) );
}
//@}

//! vec_min/max
//@{
static inline VEC_TYPE(float) sse_vec_min( const VEC_TYPE(float) a, const VEC_TYPE(float) b )
{
   return (VEC_TYPE(float))_mm_min_ps( a, b );
}
static inline VEC_TYPE(double) sse_vec_min( const VEC_TYPE(double) a, const VEC_TYPE(double) b )
{
   return (VEC_TYPE(double))_mm_min_pd( a, b );
}
static inline VEC_TYPE(int) sse_vec_min( const VEC_TYPE(int) a, const VEC_TYPE(int) b )
{
   VEC_TYPE(int) t = (VEC_TYPE(int))_mm_cmpgt_epi32((VARGTLL)a,(VARGTLL)b);
   return (VEC_TYPE(int))_mm_or_si128( _mm_and_si128((VARGTLL)t,(VARGTLL)b), _mm_andnot_si128((VARGTLL)t,(VARGTLL)a) );
}
static inline VEC_TYPE(short) sse_vec_min( const VEC_TYPE(short) a, const VEC_TYPE(short) b )
{
   return (VEC_TYPE(short))_mm_min_epi16( (VARGTLL)a, (VARGTLL)b );
}
static inline VEC_TYPE(char) sse_vec_min( const VEC_TYPE(char) a, const VEC_TYPE(char) b )
{
   VEC_TYPE(char) t = (VEC_TYPE(char))_mm_cmpgt_epi8((VARGTLL)a,(VARGTLL)b);
   return (VEC_TYPE(char))_mm_or_si128( _mm_and_si128((VARGTLL)t,(VARGTLL)b), _mm_andnot_si128((VARGTLL)t,(VARGTLL)a) );
}

static inline VEC_TYPE(float) sse_vec_max( const VEC_TYPE(float) a, const VEC_TYPE(float) b )
{
   return (VEC_TYPE(float))_mm_max_ps( a, b );
}
static inline VEC_TYPE(double) sse_vec_max( const VEC_TYPE(double) a, const VEC_TYPE(double) b )
{
   return (VEC_TYPE(double))_mm_max_pd( a, b );
}
static inline VEC_TYPE(int) sse_vec_max( const VEC_TYPE(int) a, const VEC_TYPE(int) b )
{
   VEC_TYPE(int) t = (VEC_TYPE(int))_mm_cmpgt_epi32((VARGTLL)a,(VARGTLL)b);
   return (VEC_TYPE(int))_mm_or_si128( _mm_andnot_si128((VARGTLL)t,(VARGTLL)b), _mm_and_si128((VARGTLL)t,(VARGTLL)a) );
}
static inline VEC_TYPE(short) sse_vec_max( const VEC_TYPE(short) a, const VEC_TYPE(short) b )
{
   return (VEC_TYPE(short))_mm_max_epi16( (VARGTLL)a, (VARGTLL)b );
}
static inline VEC_TYPE(char) sse_vec_max( const VEC_TYPE(char) a, const VEC_TYPE(char) b )
{
   VEC_TYPE(char) t = (VEC_TYPE(char))_mm_cmpgt_epi8((VARGTLL)a,(VARGTLL)b);
   return (VEC_TYPE(char))_mm_or_si128( _mm_andnot_si128((VARGTLL)t,(VARGTLL)b), _mm_and_si128((VARGTLL)t,(VARGTLL)a) );
}
//@}

//! movemask
//@{
static inline int sse_movemask( const VEC_TYPE(double) a )
{
   return _mm_movemask_pd( a );
}
static inline int sse_movemask( const VEC_TYPE(float) a )
{
   return _mm_movemask_ps( a );
}
static inline int sse_movemask( const VEC_TYPE(int) a )
{
   return _mm_movemask_epi8( (VARGTLL)a );
}
static inline int sse_movemask( const VEC_TYPE(short) a )
{
   return _mm_movemask_epi8( (VARGTLL)a );
}
static inline int sse_movemask( const VEC_TYPE(char) a )
{
   return _mm_movemask_epi8( (VARGTLL)a );
}
//@}


/*!
 *  implemetation of some missing math functions using SSE primitives
 */
//! unary minus
//@{
static inline VEC_TYPE(float) sse_neg( const VEC_TYPE(float) a )
{
   return (VEC_TYPE(float)) _mm_sub_ps( VEC_ZERO(float), a );
}
static inline VEC_TYPE(double) sse_neg( const VEC_TYPE(double) a )
{
   return (VEC_TYPE(double)) _mm_sub_pd( VEC_ZERO(double), a );
}
static inline VEC_TYPE(int) sse_neg( const VEC_TYPE(int) a )
{
   return (VEC_TYPE(int)) _mm_sub_epi32( (VARGTLL)VEC_ZERO(int), (VARGTLL)a );
}
static inline VEC_TYPE(short) sse_neg( const VEC_TYPE(short) a )
{
   return (VEC_TYPE(short)) _mm_sub_epi16( (VARGTLL)VEC_ZERO(short), (VARGTLL)a );
}
static inline VEC_TYPE(char) sse_neg( const VEC_TYPE(char) a )
{
   return (VEC_TYPE(char)) _mm_sub_epi8( (VARGTLL)VEC_ZERO(char), (VARGTLL)a );
}
//@}

//! fabs
//@{
static inline VEC_TYPE(float) sse_fabs( const VEC_TYPE(float) a )
{
   return (VEC_TYPE(float)) _mm_srli_epi32( _mm_slli_epi32( (VARGTLL)a, 1 ), 1 );
}
static inline VEC_TYPE(double) sse_fabs( const VEC_TYPE(double) a )
{
   return (VEC_TYPE(double)) _mm_srli_epi64( _mm_slli_epi64( (VARGTLL)a, 1 ), 1 );
}
//@}

//! square root
//@{
static inline VEC_TYPE(float) sse_sqrt( const VEC_TYPE(float) a )
{
   return (VEC_TYPE(float)) _mm_sqrt_ps( a );
}
static inline VEC_TYPE(double) sse_sqrt( const VEC_TYPE(double) a )
{
   return (VEC_TYPE(double)) _mm_sqrt_pd( a );
}
//@}

#ifdef __SSE4_1__
//! round to nearest int
//@{
static inline VEC_TYPE(float) sse_vec_round( const VEC_TYPE(float) a )
{
   // _MM_FROUND_TO_NEAREST_INT is 0x0
   return (VEC_TYPE(float))_mm_round_ps( a, 0 );
}
static inline VEC_TYPE(double) sse_vec_round( const VEC_TYPE(double) a )
{
   // _MM_FROUND_TO_NEAREST_INT is 0x0
   return (VEC_TYPE(double))_mm_round_pd( a, 0 );
}
//@}
#endif

/*!
 *  implemetation of fast integer powers up to 8 for float
 *  and double
 */
//@{
static inline VEC_TYPE(float) sse_pow2( const VEC_TYPE(float) a )
{
   return _mm_mul_ps(a,a);
}
static inline VEC_TYPE(float) sse_pow3( const VEC_TYPE(float) a )
{
   return _mm_mul_ps(a,_mm_mul_ps(a,a));
}
static inline VEC_TYPE(float) sse_pow4( const VEC_TYPE(float) a )
{
   VEC_TYPE(float) tmp2 = _mm_mul_ps(a,a);
   return _mm_mul_ps(tmp2,tmp2);
}
static inline VEC_TYPE(float) sse_pow5( const VEC_TYPE(float) a )
{
   VEC_TYPE(float) tmp2 = _mm_mul_ps(a,a);
   VEC_TYPE(float) tmp4 = _mm_mul_ps(tmp2,tmp2);
   return _mm_mul_ps(tmp4,a);
}
static inline VEC_TYPE(float) sse_pow6( const VEC_TYPE(float) a )
{
   VEC_TYPE(float) tmp2 = _mm_mul_ps(a,a);
   VEC_TYPE(float) tmp4 = _mm_mul_ps(tmp2,tmp2);
   return _mm_mul_ps(tmp2,tmp4);
}
static inline VEC_TYPE(float) sse_pow7( const VEC_TYPE(float) a )
{
   VEC_TYPE(float) tmp2 = _mm_mul_ps(a,a);
   VEC_TYPE(float) tmp4 = _mm_mul_ps(tmp2,tmp2);
   VEC_TYPE(float) tmp6 = _mm_mul_ps(tmp2,tmp4);
   return _mm_mul_ps(a,tmp6);
}
static inline VEC_TYPE(float) sse_pow8( const VEC_TYPE(float) a )
{
   VEC_TYPE(float) tmp2 = _mm_mul_ps(a,a);
   VEC_TYPE(float) tmp4 = _mm_mul_ps(tmp2,tmp2);
   return _mm_mul_ps(tmp4,tmp4);
}
static inline VEC_TYPE(double) sse_pow2( const VEC_TYPE(double) a )
{
   return _mm_mul_pd(a,a);
}
static inline VEC_TYPE(double) sse_pow3( const VEC_TYPE(double) a )
{
   return _mm_mul_pd(a,_mm_mul_pd(a,a));
}
static inline VEC_TYPE(double) sse_pow4( const VEC_TYPE(double) a )
{
   VEC_TYPE(double) tmp2 = _mm_mul_pd(a,a);
   return _mm_mul_pd(tmp2,tmp2);
}
static inline VEC_TYPE(double) sse_pow5( const VEC_TYPE(double) a )
{
   VEC_TYPE(double) tmp2 = _mm_mul_pd(a,a);
   VEC_TYPE(double) tmp4 = _mm_mul_pd(tmp2,tmp2);
   return _mm_mul_pd(tmp4,a);
}
static inline VEC_TYPE(double) sse_pow6( const VEC_TYPE(double) a )
{
   VEC_TYPE(double) tmp2 = _mm_mul_pd(a,a);
   VEC_TYPE(double) tmp4 = _mm_mul_pd(tmp2,tmp2);
   return _mm_mul_pd(tmp2,tmp4);
}
static inline VEC_TYPE(double) sse_pow7( const VEC_TYPE(double) a )
{
   VEC_TYPE(double) tmp2 = _mm_mul_pd(a,a);
   VEC_TYPE(double) tmp4 = _mm_mul_pd(tmp2,tmp2);
   VEC_TYPE(double) tmp6 = _mm_mul_pd(tmp2,tmp4);
   return _mm_mul_pd(a,tmp6);
}
static inline VEC_TYPE(double) sse_pow8( const VEC_TYPE(double) a )
{
   VEC_TYPE(double) tmp2 = _mm_mul_pd(a,a);
   VEC_TYPE(double) tmp4 = _mm_mul_pd(tmp2,tmp2);
   return _mm_mul_pd(tmp4,tmp4);
}
//@}

/*!
 *  Applicative templates for vectorized binary operators returning the
 *  c-style promoted type of their inputs.
 *  These are specializations of the applicative templates used for scalar
 *  processing. If a specialization for the given operation and its
 *  argument types exists, the expression can be vectorized.
 */

/*!
 *  applicative templates for vector unary operators
 */
#define MAKE_UNAP_VEC(classname,type,op,vec_op)                         \
template<>                                                              \
struct classname<type> : public _et_applic_base                         \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef type value_type;                                             \
   typedef type vec_value_type;                                         \
   static inline type eval( const type& a )                             \
     { return op(a); }                                                  \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a )    \
     { return (VEC_TYPE(type))vec_op(a); }                              \
}

/*!
 *  applicative templates for vector unary operators
 *  using an SSE intrinsic
 */
#define MAKE_UNAP_VEC_INTRIN(classname,type,op,vec_op)			\
template<>                                                              \
struct classname<type> : public _et_applic_base                         \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef type value_type;                                             \
   typedef type vec_value_type;                                         \
   static inline type eval( const type& a )                             \
     { return op(a); }                                                  \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a )    \
     { return (VEC_TYPE(type))vec_op((VARGT(type))a); }                 \
}

/*!
 *   applicative templates for unary functions
 */
#define MAKE_UNAP_FUNC_VEC(classname,type,op,vec_op)			\
template<>                                                              \
struct classname<type> : public _et_applic_base                         \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef type value_type;                                             \
   typedef type vec_value_type;                                         \
   static inline type eval( const type& a )                             \
     { return op(a); }                                                  \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a )    \
     { return vec_op(a); }                                              \
}

/*!
 *  applicative template for binary operators
 */
#define MAKE_BINAP_VEC(classname,type,op,vec_op)                        \
template<>                                                              \
struct classname<type,type> : public _et_applic_base                    \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef promotion_trait<type,type>::PType value_type;                \
   typedef type vec_value_type;                                         \
   static inline value_type eval( const type& a, const type& b )        \
     { return a op b; }                                                 \
                                                                        \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a,     \
                                          const VEC_TYPE(type) & b )    \
     { return (VEC_TYPE(type))vec_op(a,b); }                            \
}

/*!
 *  applicative templates for vector binary operators
 *  using an SSE intrinsic
 */
#define MAKE_BINAP_VEC_INTRIN(classname,type,op,vec_op)                 \
template<>                                                              \
struct classname<type,type> : public _et_applic_base                    \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef promotion_trait<type,type>::PType value_type;                \
   typedef type vec_value_type;                                         \
   static inline value_type eval( const type& a, const type& b )        \
     { return a op b; }                                                 \
                                                                        \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a,     \
                                          const VEC_TYPE(type) & b )    \
     { return (VEC_TYPE(type))vec_op((VARGT(type))a,(VARGT(type))b); }  \
}

/*!
 *  applicative templates for vector binary functions
 */
#define MAKE_BINAP_FUNC_VEC(classname,type,op,vec_op)                   \
template<>                                                              \
struct classname<type,type> : public _et_applic_base                    \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef promotion_trait<type,type>::PType value_type;                \
   typedef type vec_value_type;                                         \
   static inline value_type eval( const type& a, const type& b )        \
   { return op(a,b); }                                                  \
                                                                        \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a,     \
                                          const VEC_TYPE(type) & b )    \
     { return (VEC_TYPE(type))vec_op(a,b); }                            \
}



// unfortunately, MMX/SSE/SSE2/SSE3 define much fewer operations
// compared to Altivec, but we can at least use those.
//    TODO: We should try and implement more operations in terms of
//          these primitives
MAKE_UNAP_VEC(__ltl_TMinus, float  , -, sse_neg);
MAKE_UNAP_VEC(__ltl_TMinus, double , -, sse_neg);
MAKE_UNAP_VEC(__ltl_TMinus, int    , -, sse_neg);
MAKE_UNAP_VEC(__ltl_TMinus, short  , -, sse_neg);
MAKE_UNAP_VEC(__ltl_TMinus, char   , -, sse_neg);

MAKE_UNAP_VEC(__ltl_TNot, char     , !, sse_vec_not);
MAKE_UNAP_VEC(__ltl_TNot, short    , !, sse_vec_not);
MAKE_UNAP_VEC(__ltl_TNot, int      , !, sse_vec_not);
MAKE_UNAP_VEC(__ltl_TNot, long long, !, sse_vec_not);
MAKE_UNAP_VEC(__ltl_TNot, float    , !, sse_vec_not);
MAKE_UNAP_VEC(__ltl_TNot, double   , !, sse_vec_not);

MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, char     , +, _mm_add_epi8);
MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, short    , +, _mm_add_epi16);
MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, int      , +, _mm_add_epi32);
MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, long long, +, _mm_add_epi64);
MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, float    , +, _mm_add_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TAdd, double   , +, _mm_add_pd);

MAKE_BINAP_VEC_INTRIN(__ltl_TSub, char     , -, _mm_sub_epi8);
MAKE_BINAP_VEC_INTRIN(__ltl_TSub, short    , -, _mm_sub_epi16);
MAKE_BINAP_VEC_INTRIN(__ltl_TSub, int      , -, _mm_sub_epi32);
MAKE_BINAP_VEC_INTRIN(__ltl_TSub, long long, -, _mm_sub_epi64);
MAKE_BINAP_VEC_INTRIN(__ltl_TSub, float    , -, _mm_sub_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TSub, double   , -, _mm_sub_pd);

MAKE_BINAP_VEC_INTRIN(__ltl_TMul, float  , *, _mm_mul_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TMul, double , *, _mm_mul_pd);
#ifdef __SSE4_1__
MAKE_BINAP_VEC_INTRIN(__ltl_TMul, int , *, _mm_mullo_epi32);
#endif

MAKE_BINAP_VEC_INTRIN(__ltl_TDiv, float  , /, _mm_div_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TDiv, double , /, _mm_div_pd);

MAKE_BINAP_VEC_INTRIN(__ltl_TEQ , char  , ==, _mm_cmpeq_epi8);
MAKE_BINAP_VEC_INTRIN(__ltl_TEQ , short , ==, _mm_cmpeq_epi16);
MAKE_BINAP_VEC_INTRIN(__ltl_TEQ , int   , ==, _mm_cmpeq_epi32);
MAKE_BINAP_VEC_INTRIN(__ltl_TEQ , float , ==, _mm_cmpeq_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TEQ , double, ==, _mm_cmpeq_pd);

MAKE_BINAP_VEC(__ltl_TNE , char ,  !=, sse_vec_cmpneq_epi8);
MAKE_BINAP_VEC(__ltl_TNE , short,  !=, sse_vec_cmpneq_epi16);
MAKE_BINAP_VEC(__ltl_TNE , int  ,  !=, sse_vec_cmpneq_epi32);
MAKE_BINAP_VEC_INTRIN(__ltl_TNE , float , !=, _mm_cmpneq_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TNE , double, !=, _mm_cmpneq_pd);

MAKE_BINAP_VEC_INTRIN(__ltl_TLT , char , <, _mm_cmplt_epi8);
MAKE_BINAP_VEC_INTRIN(__ltl_TLT , short, <, _mm_cmplt_epi16);
MAKE_BINAP_VEC_INTRIN(__ltl_TLT , int  , <, _mm_cmplt_epi32);
MAKE_BINAP_VEC_INTRIN(__ltl_TGT , char , >, _mm_cmpgt_epi8);
MAKE_BINAP_VEC_INTRIN(__ltl_TGT , short, >, _mm_cmpgt_epi16);
MAKE_BINAP_VEC_INTRIN(__ltl_TGT , int  , >, _mm_cmpgt_epi32);

MAKE_BINAP_VEC_INTRIN(__ltl_TGT , float , > , _mm_cmpgt_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TGE , float , >=, _mm_cmpge_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TLT , float , < , _mm_cmplt_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TLE , float , <=, _mm_cmple_ps);
MAKE_BINAP_VEC_INTRIN(__ltl_TGT , double, > , _mm_cmpgt_pd);
MAKE_BINAP_VEC_INTRIN(__ltl_TGE , double, >=, _mm_cmpge_pd);
MAKE_BINAP_VEC_INTRIN(__ltl_TLT , double, < , _mm_cmplt_pd);
MAKE_BINAP_VEC_INTRIN(__ltl_TLE , double, <=, _mm_cmple_pd);

MAKE_BINAP_VEC_INTRIN(__ltl_TBitAnd, char , &, _mm_and_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitAnd, short, &, _mm_and_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitAnd, int  , &, _mm_and_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitOr , char , |, _mm_or_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitOr , short, |, _mm_or_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitOr , int  , |, _mm_or_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitXor, char , ^, _mm_xor_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitXor, short, ^, _mm_xor_si128 );
MAKE_BINAP_VEC_INTRIN(__ltl_TBitXor, int  , ^, _mm_xor_si128 );

MAKE_UNAP_FUNC_VEC(__ltl_sqrt, float , std::sqrt, sse_sqrt);
MAKE_UNAP_FUNC_VEC(__ltl_sqrt, double, std::sqrt, sse_sqrt);

MAKE_UNAP_FUNC_VEC(__ltl_fabs, float , std::fabs, sse_fabs);
MAKE_UNAP_FUNC_VEC(__ltl_fabs, double, std::fabs, sse_fabs);

MAKE_UNAP_FUNC_VEC(__ltl_pow2, double, pow2, sse_pow2);
MAKE_UNAP_FUNC_VEC(__ltl_pow3, double, pow3, sse_pow3);
MAKE_UNAP_FUNC_VEC(__ltl_pow4, double, pow4, sse_pow4);
MAKE_UNAP_FUNC_VEC(__ltl_pow5, double, pow5, sse_pow5);
MAKE_UNAP_FUNC_VEC(__ltl_pow6, double, pow6, sse_pow6);
MAKE_UNAP_FUNC_VEC(__ltl_pow7, double, pow7, sse_pow7);
MAKE_UNAP_FUNC_VEC(__ltl_pow8, double, pow8, sse_pow8);
MAKE_UNAP_FUNC_VEC(__ltl_pow2, float , pow2, sse_pow2);
MAKE_UNAP_FUNC_VEC(__ltl_pow3, float , pow3, sse_pow3);
MAKE_UNAP_FUNC_VEC(__ltl_pow4, float , pow4, sse_pow4);
MAKE_UNAP_FUNC_VEC(__ltl_pow5, float , pow5, sse_pow5);
MAKE_UNAP_FUNC_VEC(__ltl_pow6, float , pow6, sse_pow6);
MAKE_UNAP_FUNC_VEC(__ltl_pow7, float , pow7, sse_pow7);
MAKE_UNAP_FUNC_VEC(__ltl_pow8, float , pow8, sse_pow8);

#ifdef HAVE_APPLE_VECLIB
// Apple provides vector versions of libm functions ;-)

MAKE_UNAP_FUNC_VEC(__ltl_sin, float , std::sin , vsinf);
MAKE_UNAP_FUNC_VEC(__ltl_cos, float , std::cos , vcosf);
MAKE_UNAP_FUNC_VEC(__ltl_tan, float , std::tan , vtanf);
MAKE_UNAP_FUNC_VEC(__ltl_asin, float, std::asin, vasinf);
MAKE_UNAP_FUNC_VEC(__ltl_acos, float, std::acos, vacosf);
MAKE_UNAP_FUNC_VEC(__ltl_atan, float, std::atan, vatanf);
MAKE_UNAP_FUNC_VEC(__ltl_sinh, float, std::sinh, vsinhf);
MAKE_UNAP_FUNC_VEC(__ltl_cosh, float, std::cosh, vcoshf);
MAKE_UNAP_FUNC_VEC(__ltl_tanh, float, std::tanh, vtanhf);

MAKE_UNAP_FUNC_VEC(__ltl_exp, float, std::exp, vexpf);
MAKE_UNAP_FUNC_VEC(__ltl_log, float, std::log, vlogf);

MAKE_BINAP_FUNC_VEC(__ltl_atan2, float, std::atan2, vatan2f);
MAKE_BINAP_FUNC_VEC(__ltl_pow,   float, std::pow,   vpowf);

#ifdef HAVE_IEEE_MATH
MAKE_UNAP_FUNC_VEC(__ltl_asinh, float, ::asinh, vasinhf);
MAKE_UNAP_FUNC_VEC(__ltl_acosh, float, ::acosh, vacoshf);
MAKE_UNAP_FUNC_VEC(__ltl_atanh, float, ::atanh, vatanhf);
MAKE_UNAP_FUNC_VEC(__ltl_expm1, float, ::expm1, vexpm1f);
MAKE_UNAP_FUNC_VEC(__ltl_log1p, float, ::log1p, vlog1pf);
#endif

#endif

#ifdef __SSE4_1__
MAKE_UNAP_VEC_INTRIN(__ltl_ceil,  float , std::ceil , _mm_ceil_ps);
MAKE_UNAP_VEC_INTRIN(__ltl_floor, float , std::floor, _mm_floor_ps);
MAKE_UNAP_VEC_INTRIN(__ltl_ceil,  double, std::ceil , _mm_ceil_pd);
MAKE_UNAP_VEC_INTRIN(__ltl_floor, double, std::floor, _mm_floor_pd);

MAKE_UNAP_FUNC_VEC(__ltl_round, float , ::round, sse_vec_round);
MAKE_UNAP_FUNC_VEC(__ltl_round, double, ::round, sse_vec_round);
#endif

}

#endif // __LTL_APPLOCOPS_SIMD_H__
