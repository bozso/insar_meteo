/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvector_ops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FVECTOR__
#error "<ltl/fvector/fvector_ops.h> must be included via <ltl/fvector.h>, never alone!"
#endif


#ifndef __LTL_FVECTOR_OPS__
#define __LTL_FVECTOR_OPS__

#include <ltl/config.h>

namespace ltl {

// -------------------------------------------------------------------- 
/*! \file fvector_ops.h
  To be able to use a single implementation for the evaluation of
  expressions for assignment AND operator X=, we need a little trick.
  We make the evaluation routine a template of the assignment operation
  which we define below in separate objects to handle each operatorX=
*/
// -------------------------------------------------------------------- 

//! Just to keep everything together in class browsers...
class _assignfv_base { };

#define MAKE_ASSIGNFV_OP(name,op)                               \
template<class X, class Y>                                      \
class name : public _assignfv_base                              \
{                                                               \
   public:                                                      \
      typedef X value_type;                                     \
      enum { static_size = 1 };                                 \
      static inline void eval( X& restrict_ x, const Y y )      \
      { x op (X)y; }                                            \
};

MAKE_ASSIGNFV_OP(fv_equ_assign,  = )
MAKE_ASSIGNFV_OP(fv_plu_assign, += )
MAKE_ASSIGNFV_OP(fv_min_assign, -= )
MAKE_ASSIGNFV_OP(fv_mul_assign, *= )
MAKE_ASSIGNFV_OP(fv_div_assign, /= )
MAKE_ASSIGNFV_OP(fv_mod_assign, %= )
MAKE_ASSIGNFV_OP(fv_xor_assign, ^= )
MAKE_ASSIGNFV_OP(fv_and_assign, &= )
MAKE_ASSIGNFV_OP(fv_bor_assign, |= )
MAKE_ASSIGNFV_OP(fv_sle_assign, <<= )
MAKE_ASSIGNFV_OP(fv_sri_assign, >>= )


// -------------------------------------------------------------------- 
// OPERATORS
// -------------------------------------------------------------------- 

// explicitly overload default operator= since gcc 2.95.X does not
// recognise the more general templates below as such
//  
template<class T, int N, int S>
inline FVector<T,N,S>& 
FVector<T,N,S>::operator=( const FVector<T,N,S>& restrict_ v )
{
  tFVLoop< FVector<T,N,S>, FVector<T,N,S>,
    fv_equ_assign<T, T>, N >::eval(*this, v);
  return *this;
}


#define MAKE_FVEXPR_ASSIGNMENT_OP(op,name)                              \
template<class T, int N, int S>                                         \
template<class Expr>                                                    \
inline FVector<T,N,S>&                                                  \
FVector<T,N,S>::operator op( const FVExprNode<Expr,N>& e )                 \
{                                                                       \
    tFVLoop< FVector<T,N,S>, FVExprNode<Expr,N>,                           \
             name<T, typename Expr::value_type>, N >::eval(*this, e);   \
    return *this;                                                       \
}

MAKE_FVEXPR_ASSIGNMENT_OP(= , fv_equ_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(+=, fv_plu_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(-=, fv_min_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(*=, fv_mul_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(/=, fv_div_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(%=, fv_mod_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(^=, fv_xor_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(&=, fv_and_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(|=, fv_bor_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(<<=,fv_sle_assign )
MAKE_FVEXPR_ASSIGNMENT_OP(>>=,fv_sri_assign )

#define MAKE_FVV_ASSIGNMENT_OP(op,name)                 \
template<class T, int N, int S>                         \
template<class T2, int S2>                              \
inline FVector<T,N,S>&                                  \
FVector<T,N,S>::operator op( const FVector<T2,N,S2>& restrict_ e )      \
{                                                       \
    tFVLoop< FVector<T,N,S>, FVector<T2,N,S2>,          \
             name<T, T2>, N >::eval(*this, e);          \
    return *this;                                       \
}


MAKE_FVV_ASSIGNMENT_OP(=,  fv_equ_assign )
MAKE_FVV_ASSIGNMENT_OP(+=, fv_plu_assign )
MAKE_FVV_ASSIGNMENT_OP(-=, fv_min_assign )
MAKE_FVV_ASSIGNMENT_OP(*=, fv_mul_assign )
MAKE_FVV_ASSIGNMENT_OP(/=, fv_div_assign )
MAKE_FVV_ASSIGNMENT_OP(%=, fv_mod_assign )
MAKE_FVV_ASSIGNMENT_OP(^=, fv_xor_assign )
MAKE_FVV_ASSIGNMENT_OP(&=, fv_and_assign )
MAKE_FVV_ASSIGNMENT_OP(|=, fv_bor_assign )
MAKE_FVV_ASSIGNMENT_OP(<<=,fv_sle_assign )
MAKE_FVV_ASSIGNMENT_OP(>>=,fv_sri_assign )


#define MAKE_FVLITERAL_ASSIGNEMNT_OP(op,name)                   \
template<class T, int N, int S>                                 \
inline FVector<T,N,S>&                                          \
FVector<T,N,S>::operator op( const T t )                        \
{                                                               \
   FVExprNode<FVExprLiteralNode<T>,N> e(t);                           \
                                                                \
   tFVLoop< FVector<T,N,S>, FVExprNode<FVExprLiteralNode<T>, N>,      \
            name<T,T>, N >::eval(*this, e );                    \
   return *this;                                                \
}

MAKE_FVLITERAL_ASSIGNEMNT_OP(+=, fv_plu_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(-=, fv_min_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(*=, fv_mul_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(/=, fv_div_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(%=, fv_mod_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(^=, fv_xor_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(&=, fv_and_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(|=, fv_bor_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(<<=,fv_sle_assign )
MAKE_FVLITERAL_ASSIGNEMNT_OP(>>=,fv_sri_assign )

}

#endif //__LTL_FVECTOR_OPS__
