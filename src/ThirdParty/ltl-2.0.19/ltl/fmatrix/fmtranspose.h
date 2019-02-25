/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmtranspose.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
 *                         Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#ifndef __LTL_FMTRANSPOSE__
#define __LTL_FMTRANSPOSE__

#include <ltl/config.h>

namespace ltl {

/*! \file fmtranspose.h
  Transpose of a matrix or matrix expression.
*/

//
//! Unary operation node for matrix transpose.
//
template<class A, class T, int M, int N>
class TMatTransposeFMExprOp : public _et_fvector_parse_base
{
   private:
      A iter1_;

   public:
      typedef T value_type;
      enum { static_size = 1 };

      inline TMatTransposeFMExprOp( const A& a )
         : iter1_(a)
      { }

      inline value_type operator()( const int i, const int j ) const
      {
         return iter1_( j, i );
      }
};


//
//! Global transpose() matrix.
//
template<class T, int N, int M>
inline
FMExprNode<TMatTransposeFMExprOp<typename FMatrix<T,M,N>::const_iterator,
                              T, N, M>, N, M>
transpose( const FMatrix<T,M,N>& m1 ) 
{
   typedef T value_type;
   typedef TMatTransposeFMExprOp<typename FMatrix<T,M,N>::const_iterator,
                                 value_type, N, M> 
      ExprT;
   
   return FMExprNode<ExprT,N,M>( ExprT(m1.begin()) );
}

//
//! Global transpose() matrix expression.
//
template<class Expr, int N, int M>
inline
FMExprNode<TMatTransposeFMExprOp<FMExprNode<Expr,M,N>,
                              typename Expr::value_type, 
                              N, M>, N, M >
transpose( const FMExprNode<Expr,M,N>& m1 ) 
{
   typedef typename Expr::value_type value_type;
   typedef TMatTransposeFMExprOp<FMExprNode<Expr,M,N>,
                                 value_type, N, M> 
      ExprT;
   
   return FMExprNode<ExprT,N,M>( ExprT(m1) );
}

}

#endif //__LTL_FMTRANSPOSE__
