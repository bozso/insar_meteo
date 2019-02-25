/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: applicops_altivec.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/misc/applicops_altivec.h> must be included after <ltl/misc/applicops.h> !"
#endif

#if !defined(__ALTIVEC__)
#error "<ltl/misc/applicops_altivec.h> needs Altivec extensions to be enabled.\nConsult your compiler manual on how to do so\nGCC needs -faltivec switch and -framework vecLib \nfor extended support on Mac OS X"
#endif

#ifndef __LTL_APPLICOPS_SIMD_H__
#define __LTL_APPLICOPS_SIMD_H__

#ifdef HAVE_APPLE_VECLIB
// Apple provides vector versions of standard lib math functions ;-)
#  include <vecLib/vfp.h>
#endif


namespace ltl {

/*! \file applicops_altivec.h

  Specializations of applicative templates using the altivec
  vector instructions on PPC
*/



//@{

/*!
   traits mapping between scalar and vector types
   for vectorizing applicops
*/
template<typename T>
struct vec_trait
{
};

#define VEC_TRAIT(type)                         \
template<> struct vec_trait<type>               \
{                                               \
      typedef vector type vec_type;             \
      static inline vec_type init( const type x)\
               { return (vec_type)(x); }        \
}

template<> struct vec_trait<double>
{
      typedef vector float vec_type;
      static inline vec_type init( const type x)
               { return (vec_type)(x); }
};

template<> struct vec_trait<bool>
{
      typedef vector bool vec_type;
      static inline vec_type init( const type x)
               { return (vec_type)(x); }
};


VEC_TRAIT(float);
VEC_TRAIT(signed int);
VEC_TRAIT(signed short);
VEC_TRAIT(signed char);
VEC_TRAIT(unsigned int);
VEC_TRAIT(unsigned short);
VEC_TRAIT(unsigned char);
#undef VEC_TRAIT

#define VEC_TYPE(type)     vec_trait<type>::vec_type
#define T_VEC_TYPE(type)   typename VEC_TYPE(type)
#define VEC_INIT(type,val) vec_trait<type>::init(val)

//@}



//@{

/*!
 *  implemetation of some missing math functions using Altivec primitives
 */
inline VEC_TYPE(float) altivec_mul( const VEC_TYPE(float) a, const VEC_TYPE(float) b )
{
   vector unsigned tmp   = vec_splat_u32(-1);
   vector float neg_zero = (vector float)vec_sl( tmp, tmp );

   return vec_madd( a, b, neg_zero );
}

inline VEC_TYPE(float) altivec_div( const VEC_TYPE(float) a, const VEC_TYPE(float) b )
{
   vector float one      = vec_ctf( vec_splat_u32(1), 0 );

   // reciprocal estimate + newton-raphson refinement
   vector float rest = vec_re( b );
   vector float recb = vec_madd( rest, vec_nmsub( rest, b, one ), rest );
   return altivec_mul( a, recb );
}

template<typename T>
inline T altivec_logical_and( const T a, const T b )
{
   T zero = (T)vec_splat_u32(0);
   return (T)vec_nor( vec_cmpeq(a, zero), vec_cmpeq(b, zero) );
}

template<typename T>
inline T altivec_logical_or( const T a, const T b )
{
   T zero = (T)vec_splat_u32(0);
   T tmp = (T)vec_and (vec_cmpeq(a, zero), vec_cmpeq(b, zero));
   return (T)vec_nor(tmp, tmp);
}

template<typename T>
inline T altivec_not( const T a )
{
   return (T)vec_nor(a, a);
}

template<typename T>
inline T altivec_cmpneq( const T a, const T b )
{
   return altivec_not((T)vec_cmpeq(a,b));
}

template<typename T>
inline T altivec_neg( const T a )
{
   T zero = (T)vec_splat_u32(0);
   return vec_sub( zero, a );
}
//@}

/*!
 *  Applicative templates for vectorized binary operators returning the
 *  c-style promoted type of their inputs.
 *  These are specializations of the applicative templates used for scalar
 *  processing. If a specialization for the given operation and its
 *  argument types exists, the expression can be vectorized.
 */
#define MAKE_BINAP_VEC(classname,op,type,vec_op)                        \
template<>                                                              \
struct classname<type,type> : public _et_applic_base                    \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef promotion_trait<type,type>::PType value_type;                \
   static inline value_type eval( const type& a, const type& b )        \
     { return a op b; }                                                 \
                                                                        \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a,     \
                                          const VEC_TYPE(type) & b )    \
     { return (VEC_TYPE(type))vec_op(a,b); }                            \
}

/*!
 *  applicative templates for vector binary functions returning the same type
 *  as their input
 */
#define MAKE_BINAP_FUNC_VEC(classname,op,type,vec_op)                   \
template<>                                                              \
struct classname<type,type> : public _et_applic_base                    \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef promotion_trait<type,type>::PType value_type;                \
   static inline value_type eval( const type& a, const type& b )        \
   { return op(a,b); }                                                  \
                                                                        \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a,     \
                                          const VEC_TYPE(type) & b )    \
     { return (VEC_TYPE(type))vec_op(a,b); }                            \
}

/*!
 *  applicative templates for vector unary operators returning the same type
 *  as their input
 */
#define MAKE_UNAP_VEC(classname,op,type,vec_op)                         \
template<>                                                              \
struct classname<type> : public _et_applic_base                         \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef type value_type;                                             \
   static inline type eval( const type& a )                             \
     { return op(a); }                                                  \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a )    \
     { return (VEC_TYPE(type))vec_op(a); }                              \
}

/*!
 *   applicative templates for unary functions
 */
#define MAKE_UNAP_FUNC_VEC(classname,type,op,vec_op)                    \
template<>                                                              \
struct classname<type> : public _et_applic_base                         \
{                                                                       \
   enum { isVectorizable = 1 };                                         \
   typedef type value_type;                                             \
   static inline type eval( const type& a )                             \
     { return op(a); }                                                  \
   static inline VEC_TYPE(type) eval_vec( const VEC_TYPE(type) & a )    \
     { return vec_op(a); }                                              \
}


MAKE_BINAP_VEC(__ltl_TAdd, +, float, vec_add);
MAKE_BINAP_VEC(__ltl_TSub, -, float, vec_sub);
MAKE_BINAP_VEC(__ltl_TMul, *, float, altivec_mul);
MAKE_BINAP_VEC(__ltl_TDiv, /, float, altivec_div);

MAKE_BINAP_VEC(__ltl_TBitAnd, &,   signed char,  vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |,   signed char,  vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^,   signed char,  vec_xor);
MAKE_BINAP_VEC(__ltl_TBitAnd, &, unsigned char,  vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |, unsigned char,  vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^, unsigned char,  vec_xor);
MAKE_BINAP_VEC(__ltl_TBitAnd, &,   signed short, vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |,   signed short, vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^,   signed short, vec_xor);
MAKE_BINAP_VEC(__ltl_TBitAnd, &, unsigned short, vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |, unsigned short, vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^, unsigned short, vec_xor);
MAKE_BINAP_VEC(__ltl_TBitAnd, &,   signed int,   vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |,   signed int,   vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^,   signed int,   vec_xor);
MAKE_BINAP_VEC(__ltl_TBitAnd, &, unsigned int,   vec_and);
MAKE_BINAP_VEC(__ltl_TBitOr , |, unsigned int,   vec_or);
MAKE_BINAP_VEC(__ltl_TBitXor, ^, unsigned int,   vec_xor);

MAKE_BINAP_VEC(__ltl_TAnd, &&,   signed char,  altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&, unsigned char,  altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&,   signed short, altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&, unsigned short, altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&,   signed int,   altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&, unsigned int,   altivec_logical_and);
MAKE_BINAP_VEC(__ltl_TAnd, &&,         float,  altivec_logical_and);

MAKE_BINAP_VEC(__ltl_TOr , ||,   signed char,  altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||, unsigned char,  altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||,   signed short, altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||, unsigned short, altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||,   signed int,   altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||, unsigned int,   altivec_logical_or);
MAKE_BINAP_VEC(__ltl_TOr , ||,         float,  altivec_logical_or);

MAKE_BINAP_VEC(__ltl_TLT , <,   signed char,  vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <, unsigned char,  vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <,   signed short, vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <, unsigned short, vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <,   signed int,   vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <, unsigned int,   vec_cmplt);
MAKE_BINAP_VEC(__ltl_TLT , <,         float,  vec_cmplt);

MAKE_BINAP_VEC(__ltl_TGT , >,   signed char,  vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >, unsigned char,  vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >,   signed short, vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >, unsigned short, vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >,   signed int,   vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >, unsigned int,   vec_cmpgt);
MAKE_BINAP_VEC(__ltl_TGT , >,         float,  vec_cmpgt);

MAKE_BINAP_VEC(__ltl_TEQ , ==,   signed char,  vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==, unsigned char,  vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==,   signed short, vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==, unsigned short, vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==,   signed int,   vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==, unsigned int,   vec_cmpeq);
MAKE_BINAP_VEC(__ltl_TEQ , ==,         float,  vec_cmpeq);

MAKE_BINAP_VEC(__ltl_TNE , !=,   signed char,  altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=, unsigned char,  altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=,   signed short, altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=, unsigned short, altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=,   signed int,   altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=, unsigned int,   altivec_cmpneq);
MAKE_BINAP_VEC(__ltl_TNE , !=,         float,  altivec_cmpneq);

MAKE_BINAP_VEC(__ltl_TGE , >=,        float,  vec_cmpge);
MAKE_BINAP_VEC(__ltl_TLE , <=,        float,  vec_cmple);

MAKE_UNAP_VEC(__ltl_TMinus , -,   signed char,  altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -, unsigned char,  altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -,   signed short, altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -, unsigned short, altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -,   signed int,   altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -, unsigned int,   altivec_neg);
MAKE_UNAP_VEC(__ltl_TMinus , -,         float,  altivec_neg);

MAKE_UNAP_VEC(__ltl_TNot , !,   signed char,  altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !, unsigned char,  altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !,   signed short, altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !, unsigned short, altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !,   signed int,   altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !, unsigned int,   altivec_not);
MAKE_UNAP_VEC(__ltl_TNot , !,         float,  altivec_not);

//MAKE_UNAP_FUNC_VEC(__ltl_round, float, std::round, vec_round);
//MAKE_UNAP_FUNC_VEC(__ltl_trunc, float, std::trunc, vec_truc);
MAKE_UNAP_FUNC_VEC(__ltl_ceil, float, std::ceil, vec_ceil);
MAKE_UNAP_FUNC_VEC(__ltl_floor, float, std::floor, vec_floor);
MAKE_UNAP_FUNC_VEC(__ltl_fabs, float, std::fabs, vec_abs);

#ifdef HAVE_APPLE_VECLIB
// Apple provides vector versions of standard lib math functions ;-)
MAKE_UNAP_FUNC_VEC(__ltl_sqrt, float, std::sqrt, vsqrtf);
MAKE_UNAP_FUNC_VEC(__ltl_sin, float, std::sin, vsinf);
MAKE_UNAP_FUNC_VEC(__ltl_cos, float, std::cos, vcosf);
MAKE_UNAP_FUNC_VEC(__ltl_tan, float, std::tan, vtanf);
MAKE_UNAP_FUNC_VEC(__ltl_asin, float, std::asin, vasinf);
MAKE_UNAP_FUNC_VEC(__ltl_acos, float, std::acos, vacosf);
MAKE_UNAP_FUNC_VEC(__ltl_atan, float, std::atan, vatanf);
MAKE_UNAP_FUNC_VEC(__ltl_sinh, float, std::sinh, vsinhf);
MAKE_UNAP_FUNC_VEC(__ltl_cosh, float, std::cosh, vcoshf);
MAKE_UNAP_FUNC_VEC(__ltl_tanh, float, std::tanh, vtanhf);
MAKE_UNAP_FUNC_VEC(__ltl_exp, float, std::exp, vexpf);
MAKE_UNAP_FUNC_VEC(__ltl_log, float, std::log, vlogf);

MAKE_BINAP_FUNC_VEC(__ltl_atan2, std::atan2, float, vatan2f);
MAKE_BINAP_FUNC_VEC(__ltl_pow,   std::pow, float, vpowf);
#endif

}

#endif // __LTL_APPLOCOPS_SIMD_H__
