/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmatvec.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FMATVEC__
#define __LTL_FMATVEC__

#include <ltl/config.h>

namespace ltl
{

//
// template metaprogram - matrix vector products
//

// dot product returns the sum_type of the c-promoted type of the operands
#define DOT_PROMOTE(T1_,T2_)   \
    typename sumtype_trait<typename promotion_trait<T1_,T2_>::PType>::SumType


template<class A, class B, class T, int N>
class tMatVecLoop;

template<class A, class B, class T, int N, int J>
class tNMatVecLoop;

template<class A, class B, class T, int N, bool unroll>
class tMatVecSplitLoop;

// ==================================================================

//
// binary operation node for matrix vector product
//
template<class A, class B, class T, int M, int N>
class TMatVecFVExprOp : public _et_fvector_parse_base
{
   private:
      A iter1_;
      B iter2_;

   public:
      typedef T value_type;
      enum { static_size = N*(A::static_size + B::static_size) };

      inline TMatVecFVExprOp( const A& a, const B& b )
         : iter1_(a), iter2_(b)
      { }

      inline value_type operator[]( const int i ) const
      {
         return tMatVecLoop<A,B,T,N>::eval( iter1_, iter2_, i );
      }
};


//
// global dot()
// matrix vector
//
template<class T1, class T2, int M, int N, int S>
inline
FVExprNode<TMatVecFVExprOp<typename FMatrix<T1,M,N>::const_iterator,
                        typename FVector<T2,N,S>::const_iterator,
                        DOT_PROMOTE(T1,T2), M, N>,M>
dot( const FMatrix<T1,M,N>& m, const FVector<T2,N,S>& v ) 
{
   typedef DOT_PROMOTE(T1,T2) value_type;
   typedef TMatVecFVExprOp<typename FMatrix<T1,M,N>::const_iterator,
                           typename FVector<T2,N,S>::const_iterator,
                           value_type, M, N> 
      ExprT;
   
   return FVExprNode<ExprT,M>( ExprT(m.begin(), v.begin()) );
}

// matrix vecexpr
template<class T1, class A, int M, int N>
inline 
FVExprNode<TMatVecFVExprOp<typename FMatrix<T1,M,N>::const_iterator,
                        FVExprNode<A,N>,
                        DOT_PROMOTE(T1,typename A::value_type), M, N>,M>
dot( const FMatrix<T1,M,N>& m, const FVExprNode<A,N>& v ) 
{
   typedef DOT_PROMOTE(T1,typename A::value_type) value_type;
   typedef TMatVecFVExprOp<typename FMatrix<T1,M,N>::const_iterator,
                           FVExprNode<A,N>,
                           value_type, M, N>
      ExprT;
   
   return FVExprNode<ExprT,M>( ExprT(m.begin(), v ) );
}

// matrixexpr vector
template<class A, class T2, int M, int N, int S>
inline 
FVExprNode<TMatVecFVExprOp<FMExprNode<A,M,N>,
                        typename FVector<T2,N,S>::const_iterator,
                        DOT_PROMOTE(typename A::value_type,T2), M, N>, M>
dot( const FMExprNode<A,M,N>& m, const FVector<T2,N,S>& v ) 
{
   typedef DOT_PROMOTE(typename A::value_type,T2) value_type;
   typedef TMatVecFVExprOp<FMExprNode<A,M,N>,
                           typename FVector<T2,N,S>::const_iterator,
                           value_type, M, N>
      ExprT;
   
   return FVExprNode<ExprT,M>( ExprT(m, v.begin()) );
}

// matrixexpr vectorexpr
template<class A, class B, int M, int N>
inline 
FVExprNode<TMatVecFVExprOp<FMExprNode<A,M,N>,
                        FVExprNode<B,N>,
                        DOT_PROMOTE(typename A::value_type,typename B::value_type)
                        , M, N>, M>
dot( const FMExprNode<A,M,N>& m, const FVExprNode<B,N>& v ) 
{
   typedef DOT_PROMOTE(typename A::value_type,typename B::value_type) value_type;
   typedef TMatVecFVExprOp<FMExprNode<A,M,N>,
                           FVExprNode<B,N>,
                           value_type, M, N>
      ExprT;
   
   return FVExprNode<ExprT,M>( ExprT(m, v) );
}




// ==================================================================

template<class A, class B, class T, int N>
class tMatVecLoop
{
   public:
      typedef T value_type;

     enum { static_size = N*(A::static_size + B::static_size) };
   
      // for per element operations which store their result to A
      inline static value_type eval( const A& a, const B& b, const int i )
      {
         return tMatVecSplitLoop<A, B, T, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::eval(a, b, i);
      }
};


// ==================================================================

template<class A, class B, class T, int N, bool unroll>
class tMatVecSplitLoop
{ };

// template loop unrolling
//
template<class A, class B, class T, int N>
class tMatVecSplitLoop<A, B, T, N, true>
{
   public:
      typedef T value_type;
      
      inline static value_type eval( const A& a, const B& b, const int i )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNMatVecLoop<A, B, T, N, N-1>::eval( a, b, i );
      }
};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class A, class B, class T, int N>
class tMatVecSplitLoop<A, B, T, N, false>
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, const int i )
      {
         value_type tmp = value_type(0);
         for(int j=0; j<N; ++j) 
            tmp += a(i+1,j+1) * b[j];

         return tmp;
      }
};


// ==================================================================

template<class A, class B, class T, int N, int J>
class tNMatVecLoop
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, const int i )
      {
         return tNMatVecLoop<A, B, T, N, J-1 >::eval( a, b, i ) +
            a(i+1,J+1) * b[J];
      }
};


// end of recursion
template<class A, class B, class T, int N>
class tNMatVecLoop<A, B, T, N, 0>
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, const int i )
      {
         return a(i+1,1) * b[0];
      }
};


#undef DOT_PROMOTE

}

#endif //__LTL_FMATVEC__
