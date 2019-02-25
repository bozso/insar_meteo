/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvdot.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FVDOT__
#define __LTL_FVDOT__

#include <ltl/config.h>

namespace ltl {

/*! \file fvdot.h
  Inner product for ltl::FVector and ltl::FVExprNode.
*/

// dot product returns the sum_type of the c-promoted type of the operands
#define DOT_PROMOTE(T1_,T2_) \
            typename         \
            sumtype_trait<typename promotion_trait<T1_,T2_>::PType>::SumType


template<class Expr, int N>
class tVAccumLoop;
template<class Expr, int N, bool unroll>
class tVAccumSplitLoop;
template<class Expr, int N>
class tNAccumLoop;

/*
template<class Expr, int N>
typename sumtype_trait<typename Expr::value_type>::SumType
sum(  FVExprNode<Expr,N> e );

template<class T, int N, int S>
typename sumtype_trait<T>::SumType
sum(  FVector<T,N,S>& e );
*/

template<class T1, class T2, int N, int S1, int S2>
inline DOT_PROMOTE(T1,T2) dot( const FVector<T1,N,S1>& a, const FVector<T2,N,S2>& b )
{
   return sum( a*b );
}

template<class T1, class T2, int N, int S1>
inline DOT_PROMOTE(T1,T2) dot( const FVector<T1,N,S1>& a, const FVExprNode<T2,N>& b )
{
   return sum( a*b );
}

template<class T1, class T2, int N, int S2>
inline DOT_PROMOTE(T1,T2) dot( FVExprNode<T1,N> a, const FVector<T2,N,S2>& b )
{
   return sum( a*b );
}

template<class T1, class T2, int N>
inline DOT_PROMOTE(T1,T2) dot( const FVExprNode<T1,N>& a, const FVExprNode<T2,N>& b )
{
   return sum( a*b );
}



template<class Expr, int N>
inline typename sumtype_trait<typename Expr::value_type>::SumType
sum( const FVExprNode<Expr,N>& e )
{
   return tVAccumLoop< FVExprNode<Expr,N>, N >::sum(e);
}

template<class T, int N, int S>
inline typename sumtype_trait<T>::SumType
sum( const FVector<T,N,S>& e )
{
   return tVAccumLoop< FVector<T,N,S>, N >::sum(e);
}


template<class Expr, int N>
inline typename sumtype_trait<typename Expr::value_type>::SumType
sum2( const FVExprNode<Expr,N>& e )
{
   return tVAccumLoop< FVExprNode<Expr,N>, N >::sum(pow2(e));
}

template<class T, int N, int S>
inline typename sumtype_trait<T>::SumType
sum2( const FVector<T,N,S>& e )
{
   return tVAccumLoop< FVector<T,N,S>, N >::sum(pow2(e));
}

template<class Expr, int N>
inline typename sumtype_trait<typename Expr::value_type>::SumType
product( const FVExprNode<Expr,N>& e )
{
   return tVAccumLoop< FVExprNode<Expr,N>, N >::product(e);
}

template<class T, int N, int S>
inline typename sumtype_trait<T>::SumType
product( const FVector<T,N,S>& e )
{
   return tVAccumLoop< FVector<T,N,S>, N >::product(e);
}

// ======================================================================

template<class Expr, int N>
class tVAccumLoop
{
   public:
      typedef typename sumtype_trait<typename Expr::value_type>::SumType value_type;
      enum { static_size = N * Expr::static_size };
   
      inline static value_type sum( const Expr& e )
      {
         return tVAccumSplitLoop<Expr, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::sum(e);
      }

      inline static value_type product( const Expr& e )
      {
         return tVAccumSplitLoop<Expr, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::product(e);
      }
};



template<class Expr, int N, bool unroll>
class tVAccumSplitLoop
{ };

//! Template loop unrolling.
//
template<class Expr, int N>
class tVAccumSplitLoop<Expr, N, true>
{
   public:
      typedef typename sumtype_trait<typename Expr::value_type>::SumType value_type;

      inline static value_type sum( const Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNAccumLoop<Expr, N-1>::sum(e);         
      }

      inline static value_type product( const Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNAccumLoop<Expr, N-1>::product(e);         
      }
};


//! We exceed the limit for template loop unrolling: Simple for-loop.
//
template<class Expr, int N>
class tVAccumSplitLoop<Expr, N, false>
{
   public:
      typedef typename sumtype_trait<typename Expr::value_type>::SumType value_type;

      inline static value_type sum( const Expr& e )
      {
         value_type tmp = value_type(0);
         for(int i=0; i<N; ++i) 
            tmp += e[i];
         
         return tmp;
      }

      inline static value_type product( const Expr& e )
      {
         value_type tmp = value_type(1);
         for(int i=0; i<N; ++i) 
            tmp *= e[i];
         
         return tmp;
      }

};


// ==================================================================

//
//! Template loop.
//
template<class Expr, int N>
class tNAccumLoop
{
   public:
      typedef typename sumtype_trait<typename Expr::value_type>::SumType value_type;

      inline static value_type sum( const Expr& e )
      {
         return tNAccumLoop<Expr, N-1>::sum(e) + e[N];         
      }

      inline static value_type product( const Expr& e )
      {
         return tNAccumLoop<Expr, N-1>::product(e) * e[N];         
      }
};


//! End of recursion.
template<class Expr>
class tNAccumLoop<Expr, 0>
{
   public:
      typedef typename Expr::value_type value_type;

      inline static value_type sum( const Expr& e )
      {
         return e[0];
      }

      inline static value_type product( const Expr& e )
      {
         return e[0];
      }
};

#undef DOT_PROMOTE

}

#endif //__LTL_FVDOT__
