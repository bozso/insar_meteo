/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: applicops.h 542 2014-07-09 17:03:46Z drory $
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


#if !defined(__LTL_IN_FILE_MARRAY__) && !defined(__LTL_IN_FILE_FVECTOR__) && !defined(__LTL_IN_FILE_FMATRIX__)
#error "<ltl/misc/applicops.h> must be included via ltl headers, never alone!"
#endif


#ifndef __LTL_APPLICOPS_H__
#define __LTL_APPLICOPS_H__

#include <ltl/misc/type_promote.h>

namespace ltl {

/*! \file applicops.h

  Applicative template classes for use in expression templates.

  These represent operations in the expression parse tree. They hold
  the operation that is actually performed on the elements of the 
  container, and therefore take these as arguments.
  Each applicative template has a corresponding operator
  or function (defined in expr.h for MArrays) which takes the container
  itself and/or whole expressions as arguments and are used to build 
  up the parse tree.

  These objects defined here are used by all expression template engines 
  in LTL (\c MArray, \c FVector, and \c FMatrix)

  The actual declarations of the operations on MArrays, FVectors, etc,
  are found in marray/expr_ops.h, fvector/fvector_ops.h, etc, ... 
  Only the common parts, i.e. the applicative templates are defined here. 
*/


//@{

//! This is just to keep everything together in class browsers
struct _et_applic_base
{ 
   enum { isVectorizable = 0 };
};


/*!
 *  applicative templates for binary operators returning the
 *  c-style promoted type of their inputs
 */
#define MAKE_BINAP(classname,op)                                        \
template<class T1, class T2>                                            \
struct classname : public _et_applic_base                               \
{                                                                       \
   typedef typename promotion_trait<T1,T2>::PType value_type;           \
   typedef value_type vec_value_type;                                   \
   static inline value_type eval( const T1& a, const T2& b )            \
     { return a op b; }                                                 \
}


/*!
 *  applicative templates for binary operators returning a fixed unpromoted
 *  type (eg. bool, logical operations)
 */
#define MAKE_BINAP_RET(classname,ret_type,op)                   \
template<class T1, class T2>                                    \
struct classname : public _et_applic_base                       \
{                                                               \
   typedef ret_type value_type;                                 \
   typedef value_type vec_value_type;                           \
   static inline value_type eval( const T1& a, const T2& b )    \
     { return a op b; }                                         \
}


/*!
 *  applicative templates for binary functions returning a fixed unpromoted 
 *  type (e.g. stdlib math functions)
 */
#define MAKE_BINAP_FUNC(classname,ret_type,func)                \
template<class T1, class T2>                                    \
struct classname : public _et_applic_base                       \
{                                                               \
   typedef ret_type value_type;                                 \
   typedef value_type vec_value_type;                           \
   static inline ret_type eval( const T1& a, const T2& b )      \
     { return func( a, b ); }                                   \
}

/*!
 *  applicative templates for unary operators returning the same type
 *  as their input
 */
#define MAKE_UNAP(classname,op)                                         \
template<class T>                                                       \
struct classname : public _et_applic_base                               \
{                                                                       \
   typedef T value_type;                                                \
   typedef value_type vec_value_type;                                   \
   static inline T eval( const T&  a )                                  \
     { return op(a); }                                                  \
}


/*!
 *   applicative templates for unary functions
 */
#define MAKE_UNAP_FUNC(classname,ret_type,op)                           \
template<class T>                                                       \
struct classname : public _et_applic_base                               \
{                                                                       \
   typedef ret_type value_type;                                         \
   typedef value_type vec_value_type;                                   \
   static inline ret_type eval( const T& a )                            \
     { return op(a); }                                                  \
}

/*!
 *   applicative templates for unary floating point classification functions
 */
#define MAKE_UNAP_FPC(classname,op)                                     \
template<class T>                                                       \
struct classname : public _et_applic_base                               \
{                                                                       \
   typedef int value_type;                                              \
   typedef value_type vec_value_type;                                   \
   static inline int eval( const T& )                                   \
   { return 0; }                                                        \
};                                                                      \
template<>                                                              \
struct classname <float> : public _et_applic_base                       \
{                                                                       \
   typedef int value_type;                                              \
   typedef value_type vec_value_type;                                   \
   static inline int eval( const float& a )                             \
     { return op(a); }                                                  \
};                                                                      \
template<>                                                              \
struct classname <double> : public _et_applic_base                      \
{                                                                       \
   typedef int value_type;                                              \
   typedef value_type vec_value_type;                                   \
   static inline int eval( const double& a )                            \
     { return op(a); }                                                  \
};                                                                      \
template<>                                                              \
struct classname <long double> : public _et_applic_base                 \
{                                                                       \
   typedef int value_type;                                              \
   typedef value_type vec_value_type;                                   \
   static inline int eval( const long double& a )                       \
     { return op(a); }                                                  \
}

#ifdef LTL_COMPLEX_MATH
/*!
 *  specialization for complex types \c complex<T>
 *  applicative templates for unary operators returning the same type
 *  as their input
 */
#define MAKE_UNAP_FUNC_CPL(classname,op)                                \
template<class T>                                                       \
struct classname<complex<T> > : public _et_applic_base                  \
{                                                                       \
   typedef complex<T> value_type;                                       \
   typedef value_type vec_value_type;                                   \
   static inline value_type eval( const complex<T>& a )                 \
     { return op(a); }                                                  \
}
/*!
 *  specialization for complex types \c complex<T>
 *  applicative templates for unary operators returning T
 */
#define MAKE_UNAP_FUNC_CPL_T(classname,op)                              \
template<class T>                                                       \
struct classname<complex<T> > : public _et_applic_base                  \
{                                                                       \
   typedef T value_type;                                                \
   typedef value_type vec_value_type;                                   \
   static inline value_type eval( const complex<T>& a )                 \
     { return op(a); }                                                  \
}
/*!
 *  specialization for complex types \c complex<T>
 *  applicative templates for binary functions returning the same type
 *  as input (e.g. complex stdlib math functions)
 */
#define MAKE_BINAP_FUNC_CPL(classname,func)                                  \
template<class T>                                                            \
struct classname<complex<T>,complex<T> > : public _et_applic_base            \
{                                                                            \
   typedef complex<T> value_type;                                            \
   typedef value_type vec_value_type;                                        \
   static inline value_type eval( const complex<T>& a, const complex<T>& b ) \
     { return func( a, b ); }                                                \
}

#endif


/// \cond DOXYGEN_IGNORE
/*!
 *  standard operators
 *  these are common to MArray, FVector, and FMatrix
 */
MAKE_BINAP(__ltl_TAdd, +);
MAKE_BINAP(__ltl_TSub, -);
MAKE_BINAP(__ltl_TMul, *);
MAKE_BINAP(__ltl_TDiv, /);

MAKE_BINAP(__ltl_TBitAnd, &);
MAKE_BINAP(__ltl_TBitOr , |);
MAKE_BINAP(__ltl_TBitXor, ^);
MAKE_BINAP(__ltl_TMod   , %);

MAKE_BINAP_RET(__ltl_TAnd, bool, && );
MAKE_BINAP_RET(__ltl_TOr , bool, || );
MAKE_BINAP_RET(__ltl_TGT , bool, >  );
MAKE_BINAP_RET(__ltl_TLT , bool, <  );
MAKE_BINAP_RET(__ltl_TGE , bool, >= );
MAKE_BINAP_RET(__ltl_TLE , bool, <= );
MAKE_BINAP_RET(__ltl_TNE , bool, != );
MAKE_BINAP_RET(__ltl_TEQ , bool, == );

MAKE_UNAP( __ltl_TPlus,  + );
MAKE_UNAP( __ltl_TMinus, - );
MAKE_UNAP( __ltl_TNot,   ! );
MAKE_UNAP( __ltl_TNeg,   ~ );


/*!
 *  moderatlely safe floating point comarison
 *  See Knuth, D.E., The art of computer programming, Vol II.
 *
 *  (Not sure this will work correctly in all cases involving
 *  inf, nan, or denormals)
 */
//{@
static inline bool fneq( const float a, const float b )
{
   const float ab = fabsf(a-b);
   return  (ab > 1e-7f*fabsf(a)) && (ab > 1e-7f*fabsf(b));
}
static inline bool feq( const float a, const float b )
{
   const float ab = fabsf(a-b);
   return  (ab <= 1e-7f*fabsf(a)) || (ab <= 1e-7f*fabsf(b));
}
static inline bool fneq( const double a, const double b )
{
   const double ab = fabs(a-b);
   return  (ab > 1e-15*fabs(a)) && (ab > 1e-15*fabs(b));
}
static inline bool feq( const double a, const double b )
{
   const double ab = fabs(a-b);
   return  (ab <= 1e-14*fabs(a)) || (ab <= 1e-14*fabs(b));
}
MAKE_BINAP_FUNC( __ltl_fneq, bool, fneq );
MAKE_BINAP_FUNC( __ltl_feq, bool, feq  );
//@}


/*!
 *
 * All standard library math functions
 *
 * acos        inverse trigonometric function
 * acosh       inverse hyperbolic function   
 * asin        inverse trigonometric function
 * asinh       inverse hyperbolic function   
 * atan        inverse trigonometric function
 * atanh       inverse hyperbolic function   
 * atan2       inverse trigonometric function
 * cabs        complex absolute value        
 * cbrt        cube root                     
 * ceil        integer no less than          
 * copysign    copy sign bit                 
 * cos         trigonometric function        
 * cosh        hyperbolic function           
 * erf         error function                
 * erfc        complementary error function  
 * exp         exponential                   
 * expm1       exp(x)-1                      
 * fabs        absolute value                
 * floor       integer no greater than       
 * hypot       Euclidean distance            
 * ilogb       exponent extraction           
 * infnan      signals exceptions
 * j0          bessel function               
 * j1          bessel function               
 * jn          bessel function               
 * lgamma      log gamma function; (formerly gamma.3m)
 * log         natural logarithm       
 * log10       logarithm to base 10    
 * log1p       log(1+x)                
 * pound       ronund to nearest integer
 * pow         exponential x**y
 * remainder   remainder               
 * rint        round to nearest integer
 * scalbn      exponent adjustment     
 * sin         trigonometric function  
 * sinh        hyperbolic function     
 * sqrt        square root             
 * tan         trigonometric function  
 * tanh        hyperbolic function     
 * y0          bessel function         
 * y1          bessel function
 * yn          bessel function
 * 
 */
MAKE_UNAP_FUNC( __ltl_sin,   double, std::sin   );
MAKE_UNAP_FUNC( __ltl_cos,   double, std::cos   );
MAKE_UNAP_FUNC( __ltl_tan,   double, std::tan   );
MAKE_UNAP_FUNC( __ltl_asin,  double, std::asin  );
MAKE_UNAP_FUNC( __ltl_acos,  double, std::acos  );
MAKE_UNAP_FUNC( __ltl_atan,  double, std::atan  );
MAKE_UNAP_FUNC( __ltl_sinh,  double, std::sinh  );
MAKE_UNAP_FUNC( __ltl_cosh,  double, std::cosh  );
MAKE_UNAP_FUNC( __ltl_tanh,  double, std::tanh  );
MAKE_UNAP_FUNC( __ltl_exp,   double, std::exp   );
MAKE_UNAP_FUNC( __ltl_log,   double, std::log   );
MAKE_UNAP_FUNC( __ltl_log10, double, std::log10 );
MAKE_UNAP_FUNC( __ltl_sqrt,  double, std::sqrt  );
MAKE_UNAP_FUNC( __ltl_fabs,  double, std::fabs  );
MAKE_UNAP_FUNC( __ltl_floor, double, std::floor );
MAKE_UNAP_FUNC( __ltl_ceil,  double, std::ceil  );
MAKE_UNAP_FUNC( __ltl_round, double, round );
MAKE_BINAP_FUNC( __ltl_pow,  double, std::pow   );
MAKE_BINAP_FUNC( __ltl_fmod, double, std::fmod  );
MAKE_BINAP_FUNC( __ltl_atan2,double, std::atan2 );

/*! 
   IEEE math functions
   from C99 standard
*/
#ifdef HAVE_IEEE_MATH
MAKE_UNAP_FUNC( __ltl_asinh,  double, ::asinh  );
MAKE_UNAP_FUNC( __ltl_acosh,  double, ::acosh  );
MAKE_UNAP_FUNC( __ltl_atanh,  double, ::atanh  );
MAKE_UNAP_FUNC( __ltl_cbrt,   double, ::cbrt   );
MAKE_UNAP_FUNC( __ltl_expm1,  double, ::expm1  );
MAKE_UNAP_FUNC( __ltl_log1p,  double, ::log1p  );
MAKE_UNAP_FUNC( __ltl_erf,    double, ::erf    );
MAKE_UNAP_FUNC( __ltl_erfc,   double, ::erfc   );
MAKE_UNAP_FUNC( __ltl_j0,     double, ::j0     );
MAKE_UNAP_FUNC( __ltl_j1,     double, ::j1     );
MAKE_UNAP_FUNC( __ltl_y0,     double, ::y0     );
MAKE_UNAP_FUNC( __ltl_y1,     double, ::y1     );
MAKE_UNAP_FUNC( __ltl_lgamma, double, ::lgamma );
MAKE_UNAP_FUNC( __ltl_rint,   double, ::rint   );

MAKE_BINAP_FUNC( __ltl_hypot, double, ::hypot  );

MAKE_UNAP_FPC( __ltl_fpclassify, std::fpclassify );
MAKE_UNAP_FPC( __ltl_isfinite,   std::isfinite   );
MAKE_UNAP_FPC( __ltl_isinf,      std::isinf      );
MAKE_UNAP_FPC( __ltl_isnan,      std::isnan      );
MAKE_UNAP_FPC( __ltl_isnormal,   std::isnormal   );

#endif


/*!
   Optimal implementation of pow with integer exponent <= 8
*/
inline double pow2( const double x )
{
   return x*x;
}
inline double pow3( const double x )
{
   return x*x*x;
}
inline double pow4( const double x )
{
   const double tmp = x*x;
   return tmp*tmp;
}
inline double pow5( const double x )
{
   const double tmp = x*x;
   return tmp*tmp*x;
}
inline double pow6( const double x )
{
   const double tmp = x*x*x;
   return tmp*tmp;
}
inline double pow7( const double x )
{
   const double tmp = x*x;
   return tmp*tmp*tmp*x;
}
inline double pow8( const double x )
{
   const double tmp1 = x*x;
   const double tmp2 = tmp1*tmp1;
   return tmp2*tmp2;
}
MAKE_UNAP_FUNC( __ltl_pow2, double, pow2 );
MAKE_UNAP_FUNC( __ltl_pow3, double, pow3 );
MAKE_UNAP_FUNC( __ltl_pow4, double, pow4 );
MAKE_UNAP_FUNC( __ltl_pow5, double, pow5 );
MAKE_UNAP_FUNC( __ltl_pow6, double, pow6 );
MAKE_UNAP_FUNC( __ltl_pow7, double, pow7 );
MAKE_UNAP_FUNC( __ltl_pow8, double, pow8 );


#ifdef LTL_COMPLEX_MATH
template <class T> struct __ltl_abs;
template <class T> struct __ltl_arg;
template <class T> struct __ltl_norm;
template <class T> struct __ltl_imag;
template <class T> struct __ltl_real;
template <class T> struct __ltl_conj;
MAKE_UNAP_FUNC_CPL_T( __ltl_abs, std::abs);
MAKE_UNAP_FUNC_CPL_T( __ltl_arg, std::arg);
MAKE_UNAP_FUNC_CPL_T( __ltl_norm, std::norm);
MAKE_UNAP_FUNC_CPL_T( __ltl_imag, std::imag);
MAKE_UNAP_FUNC_CPL_T( __ltl_real, std::real);
MAKE_UNAP_FUNC_CPL( __ltl_conj , std::conj);
MAKE_UNAP_FUNC_CPL( __ltl_cos  , std::cos );
MAKE_UNAP_FUNC_CPL( __ltl_cosh , std::cosh);
MAKE_UNAP_FUNC_CPL( __ltl_exp  , std::exp );
MAKE_UNAP_FUNC_CPL( __ltl_log  , std::log );
MAKE_UNAP_FUNC_CPL( __ltl_log10, std::log10);
MAKE_UNAP_FUNC_CPL( __ltl_sin  , std::sin );
MAKE_UNAP_FUNC_CPL( __ltl_sinh , std::sinh);
MAKE_UNAP_FUNC_CPL( __ltl_sqrt , std::sqrt);
MAKE_UNAP_FUNC_CPL( __ltl_tan  , std::tan );
MAKE_UNAP_FUNC_CPL( __ltl_tanh , std::tanh);
MAKE_UNAP_FUNC_CPL( __ltl_pow2, pow2 );
MAKE_UNAP_FUNC_CPL( __ltl_pow3, pow3 );
MAKE_UNAP_FUNC_CPL( __ltl_pow4, pow4 );
MAKE_UNAP_FUNC_CPL( __ltl_pow5, pow5 );
MAKE_UNAP_FUNC_CPL( __ltl_pow6, pow6 );
MAKE_UNAP_FUNC_CPL( __ltl_pow7, pow7 );
MAKE_UNAP_FUNC_CPL( __ltl_pow8, pow8 );
MAKE_BINAP_FUNC_CPL( __ltl_pow, std::pow );
/*
template<typename _Tp> complex<_Tp> polar(const _Tp&, const _Tp& = 0);
template<typename _Tp> complex<_Tp> pow(const complex<_Tp>&, const _Tp&);
template<typename _Tp> complex<_Tp> pow(const _Tp&, const complex<_Tp>&);
*/
#endif

/// \endcond

//@}

}

#endif // __LTL_APPLOCOPS_H__
