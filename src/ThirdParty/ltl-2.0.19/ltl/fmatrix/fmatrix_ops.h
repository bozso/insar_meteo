/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmatrix_ops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FMATRIX__
#error "<ltl/fmatrix/fmatrix_ops.h> must be included via <ltl/fmatrix.h>, never alone!"
#endif


#ifndef __LTL_FMATRIX_OPS__
#define __LTL_FMATRIX_OPS__

#include <ltl/config.h>

namespace ltl
{


// -------------------------------------------------------------------- 
//
// To be able to use a single implementation for the evaluation of
// expressions for assignment AND operator X=, we need a little trick.
// We make the evaluation routine a template of the assignment operation
// which we define below in separate objects to handle each operatorX=
//
// -------------------------------------------------------------------- 

// this is just to keep everything together in class browsers
class _assignfm_base { };

#define MAKE_ASSIGNFM_OP(name,op)                               \
template<class X, class Y>                                      \
class name : public _assignfm_base                              \
{                                                               \
   public:                                                      \
      typedef X value_type;                                     \
      enum { static_size = 1 };                                 \
      static inline void eval( X& restrict_ x, const Y y )      \
      { x op (X)y; }                                            \
};

MAKE_ASSIGNFM_OP(fm_equ_assign,  = )
MAKE_ASSIGNFM_OP(fm_plu_assign, += )
MAKE_ASSIGNFM_OP(fm_min_assign, -= )
MAKE_ASSIGNFM_OP(fm_mul_assign, *= )
MAKE_ASSIGNFM_OP(fm_div_assign, /= )
MAKE_ASSIGNFM_OP(fm_mod_assign, %= )
MAKE_ASSIGNFM_OP(fm_xor_assign, ^= )
MAKE_ASSIGNFM_OP(fm_and_assign, &= )
MAKE_ASSIGNFM_OP(fm_bor_assign, |= )
MAKE_ASSIGNFM_OP(fm_sle_assign, <<= )
MAKE_ASSIGNFM_OP(fm_sri_assign, >>= )


// -------------------------------------------------------------------- 
// OPERATORS
// -------------------------------------------------------------------- 

// explicitly overload default operator= since gcc 2.95.X does not
// recognise the more general templates below as such
//  
template<class T, int M, int N>
inline FMatrix<T,M,N>& 
FMatrix<T,M,N>::operator=( const FMatrix<T,M,N>& v )
{
  tFMLoop< FMatrix<T,M,N>, FMatrix<T,M,N>,
    fm_equ_assign<T, T>, M, N >::eval(*this, v);
  return *this;
}

#define MAKE_FMEXPR_ASSIGNMENT_OP(op,name)              \
template<class T, int M, int N>                         \
template<class Expr>                                    \
inline FMatrix<T,M,N>&                                  \
FMatrix<T,M,N>::operator op( const FMExprNode<Expr,M,N>& e )      \
{                                                       \
    tFMLoop< FMatrix<T,M,N>, FMExprNode<Expr,M,N>,         \
             name<T, typename Expr::value_type>,        \
             M, N >::eval( *this, e );                  \
    return *this;                                       \
}

MAKE_FMEXPR_ASSIGNMENT_OP(= , fm_equ_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(+=, fm_plu_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(-=, fm_min_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(*=, fm_mul_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(/=, fm_div_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(%=, fm_mod_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(^=, fm_xor_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(&=, fm_and_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(|=, fm_bor_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(<<=,fm_sle_assign )
MAKE_FMEXPR_ASSIGNMENT_OP(>>=,fm_sri_assign )

#define MAKE_FMM_ASSIGNMENT_OP(op,name)                 \
template<class T, int M, int N>                         \
template<class T2>                                      \
inline FMatrix<T,M,N>&                                  \
FMatrix<T,M,N>::operator op( const FMatrix<T2,M,N>& restrict_ e )       \
{                                                       \
    tFMLoop< FMatrix<T,M,N>,                            \
             typename FMatrix<T2,M,N>::const_iterator,  \
             name<T, T2>,                               \
             M, N >::eval( *this, e );                  \
    return *this;                                       \
}

MAKE_FMM_ASSIGNMENT_OP(=,  fm_equ_assign )
MAKE_FMM_ASSIGNMENT_OP(+=, fm_plu_assign )
MAKE_FMM_ASSIGNMENT_OP(-=, fm_min_assign )
MAKE_FMM_ASSIGNMENT_OP(*=, fm_mul_assign )
MAKE_FMM_ASSIGNMENT_OP(/=, fm_div_assign )
MAKE_FMM_ASSIGNMENT_OP(%=, fm_mod_assign )
MAKE_FMM_ASSIGNMENT_OP(^=, fm_xor_assign )
MAKE_FMM_ASSIGNMENT_OP(&=, fm_and_assign )
MAKE_FMM_ASSIGNMENT_OP(|=, fm_bor_assign )
MAKE_FMM_ASSIGNMENT_OP(<<=,fm_sle_assign )
MAKE_FMM_ASSIGNMENT_OP(>>=,fm_sri_assign )



#define MAKE_FMLITERAL_ASSIGNEMNT_OP(op,name)           \
template<class T, int M, int N>                         \
inline FMatrix<T,M,N>&                                  \
FMatrix<T,M,N>::operator op( const T t )                \
{                                                       \
   FMExprNode<FMExprLiteralNode<T>,M,N> e(t);                 \
                                                        \
   tFMLoop< FMatrix<T,M,N>,                             \
            FMExprNode<FMExprLiteralNode<T>,M,N>,             \
            name<T,T>, M, N >::eval( *this, e );        \
   return *this;                                        \
}

MAKE_FMLITERAL_ASSIGNEMNT_OP(+=, fm_plu_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(-=, fm_min_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(*=, fm_mul_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(/=, fm_div_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(%=, fm_mod_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(^=, fm_xor_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(&=, fm_and_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(|=, fm_bor_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(<<=,fm_sle_assign )
MAKE_FMLITERAL_ASSIGNEMNT_OP(>>=,fm_sri_assign )

}

#endif // __LTL_FMATRIX_OPS__
