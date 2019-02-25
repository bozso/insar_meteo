/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmatmat.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FMATMAT__
#define __LTL_FMATMAT__

#include <ltl/config.h>

namespace ltl
{

//
// template metaprogram - matrix matrix products
//

// dot product returns the sum_type of the c-promoted type of the operands
#define DOT_PROMOTE(T1_,T2_)   \
    typename sumtype_trait<typename promotion_trait<T1_,T2_>::PType>::SumType


template<class A, class B, class T, int N>
class tMatMatLoop;

template<class A, class B, class T, int N, int K>
class tNMatMatLoop;

template<class A, class B, class T, int N, bool unroll>
class tMatMatSplitLoop;

// ==================================================================

//
// binary operation node for matrix vector product
//
template<class A, class B, class T, int N>
class TMatMatFMExprOp : public _et_fvector_parse_base
{
   private:
      A iter1_;
      B iter2_;

   public:
      typedef T value_type;
      enum { static_size = N*(A::static_size + B::static_size) };

      inline TMatMatFMExprOp( const A& a, const B& b )
         : iter1_(a), iter2_(b)
      { }

      inline value_type operator()( const int i, const int j ) const
      {
         return tMatMatLoop<A,B,T,N>::eval( iter1_, iter2_, i, j );
      }
};


//
// global dot()
// matrix matrix
//
template<class T1, class T2, int M, int N, int K>
inline
FMExprNode<TMatMatFMExprOp<typename FMatrix<T1,M,N>::const_iterator,
                        typename FMatrix<T2,N,K>::const_iterator,
                        DOT_PROMOTE(T1,T2), N>,M,K>
dot( const FMatrix<T1,M,N>& m1, const FMatrix<T2,N,K>& m2 ) 
{
   typedef DOT_PROMOTE(T1,T2) value_type;
   typedef TMatMatFMExprOp<typename FMatrix<T1,M,N>::const_iterator,
                           typename FMatrix<T2,N,K>::const_iterator,
                           value_type, N> 
      ExprT;
   
   return FMExprNode<ExprT,M,K>( ExprT(m1.begin(), m2.begin()) );
}

// matrix matexpr
template<class T1, class A, int M, int N, int K>
inline 
FMExprNode<TMatMatFMExprOp<typename FMatrix<T1,M,N>::const_iterator,
                        FMExprNode<A,N,K>,
                        DOT_PROMOTE(T1,typename A::value_type), N>,M,K>
dot( const FMatrix<T1,M,N>& m1, const FMExprNode<A,N,K>& m2 ) 
{
   typedef DOT_PROMOTE(T1,typename A::value_type) value_type;
   typedef TMatMatFMExprOp<typename FMatrix<T1,M,N>::const_iterator,
                           FMExprNode<A,N,K>,
                           value_type, N>
      ExprT;
   
   return FMExprNode<ExprT,M,K>( ExprT(m1.begin(), m2 ) );
}

// matrixexpr matrix
template<class A, class T2, int M, int N, int K>
inline 
FMExprNode<TMatMatFMExprOp<FMExprNode<A,M,N>,
                        typename FMatrix<T2,N,K>::const_iterator,
                        DOT_PROMOTE(typename A::value_type,T2), N>, M,K>
dot( const FMExprNode<A,M,N>& m1, const FMatrix<T2,N,K>& m2 ) 
{
   typedef DOT_PROMOTE(typename A::value_type,T2) value_type;
   typedef TMatMatFMExprOp<FMExprNode<A,M,N>,
                           typename FMatrix<T2,N,K>::const_iterator,
                           value_type, N>
      ExprT;
   
   return FMExprNode<ExprT,M,K>( ExprT(m1, m2.begin()) );
}

// matrixexpr matrixexpr
template<class A, class B, int M, int N, int K>
inline 
FMExprNode<TMatMatFMExprOp<FMExprNode<A,M,N>,
                        FMExprNode<B,N,K>,
                        DOT_PROMOTE(typename A::value_type,
                                    typename B::value_type)
                        , N>, M,K>
dot( const FMExprNode<A,M,N>& m1, const FMExprNode<B,N,K>& m2 ) 
{
   typedef DOT_PROMOTE(typename A::value_type,
                       typename B::value_type) value_type;
   typedef TMatMatFMExprOp<FMExprNode<A,M,N>,
                           FMExprNode<B,N,K>,
                           value_type, N>
      ExprT;
   
   return FMExprNode<ExprT,M,K>( ExprT(m1, m2) );
}




// ==================================================================

template<class A, class B, class T, int N>
class tMatMatLoop
{
   public:
      typedef T value_type;

     enum { static_size = N*(A::static_size + B::static_size) };
   
      // for per element operations which store their result to A
      inline static value_type eval( const A& a, const B& b, 
                                     const int i, const int j )
      {
         return tMatMatSplitLoop<A, B, T, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::eval(a, b, i, j);
      }
};


// ==================================================================

template<class A, class B, class T, int N, bool unroll>
class tMatMatSplitLoop
{ };

// template loop unrolling
//
template<class A, class B, class T, int N>
class tMatMatSplitLoop<A, B, T, N, true>
{
   public:
      typedef T value_type;
      
      inline static value_type eval( const A& a, const B& b, 
                                     const int i, const int j )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNMatMatLoop<A, B, T, N, N>::eval( a, b, i, j );
      }
};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class A, class B, class T, int N>
class tMatMatSplitLoop<A, B, T, N, false>
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, 
                                     const int i, const int j )
      {
         value_type tmp = value_type(0);
         for(int k=1; k<=N; ++k) 
            tmp += a(i,k) * b(k,j);

         return tmp;
      }
};


// ==================================================================

template<class A, class B, class T, int N, int K>
class tNMatMatLoop
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, 
                                     const int i, const int j )
      {
         return tNMatMatLoop<A, B, T, N, K-1 >::eval( a, b, i, j ) +
            a(i,K) * b(K,j);
      }
};


// end of recursion
template<class A, class B, class T, int N>
class tNMatMatLoop<A, B, T, N, 1>
{
   public:
      typedef T value_type;

      inline static value_type eval( const A& a, const B& b, 
                                     const int i, const int j )
      {
         return a(i,1) * b(1,j);
      }
};


#undef DOT_PROMOTE

}

#endif //__LTL_FMATMAT__
