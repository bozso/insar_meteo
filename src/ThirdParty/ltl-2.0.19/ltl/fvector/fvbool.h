/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvbool.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FVBOOL__
#define __LTL_FVBOOL__

#include <ltl/config.h>

namespace ltl {

//
//! \file fvbool.h Template metaprograms - bool reductions for ltl::FVector.
//

template<class Expr, int N> class tVBoolLoop;
template<class Expr, int N, bool unroll> class tVBoolSplitLoop;
template<class Expr, int N> class tNBoolLoop;

template< class T, int M, int N> class FMatrix;
template< class Expr, int M, int N> class FMExprNode;


//---------------------------------------------------------------------
// AT LEAST 1 TRUE
//---------------------------------------------------------------------

template<class Expr, int N>
inline bool anyof(  const FVExprNode<Expr,N>& e )
{
   return tVBoolLoop< FVExprNode<Expr,N>, N >::anyof(const_cast<FVExprNode<Expr,N>&>(e));
}

template<class T, int N, int S>
inline bool anyof(  FVector<T,N,S>& e )
{
   return tVBoolLoop< FVector<T,N,S>, N >::anyof(e);
}

//---------------------------------------------------------------------
// ALL TRUE
//---------------------------------------------------------------------

template<class Expr, int N>
inline bool allof(  const FVExprNode<Expr,N>& e )
{
   return tVBoolLoop< FVExprNode<Expr,N>, N >::allof(const_cast<FVExprNode<Expr,N>&>(e));
}

template<class T, int N, int S>
inline bool allof(  FVector<T,N,S>& e )
{
   return tVBoolLoop< FVector<T,N,S>, N >::allof(e);
}


//---------------------------------------------------------------------
// NONE TRUE
//---------------------------------------------------------------------

template<class Expr, int N>
inline bool noneof(  const FVExprNode<Expr,N>& e )
{
   return tVBoolLoop< FVExprNode<Expr,N>, N >::noneof(const_cast<FVExprNode<Expr,N>&>(e));
}

template<class T, int N, int S>
inline bool noneof(  FVector<T,N,S>& e )
{
   return tVBoolLoop< FVector<T,N,S>, N >::noneof(e);
}


// ======================================================================

template<class Expr, int N>
class tVBoolLoop
{
   public:
      typedef bool value_type;
      enum { static_size = N * Expr::static_size };
   
      inline static bool anyof( Expr& e )
      {
         return tVBoolSplitLoop<Expr, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::anyof(e);
      }
      inline static bool allof( Expr& e )
      {
         return tVBoolSplitLoop<Expr, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::allof(e);
      }
      inline static bool noneof( Expr& e )
      {
         return tVBoolSplitLoop<Expr, N,
                                (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::noneof(e);
      }
};



template<class Expr, int N, bool unroll>
class tVBoolSplitLoop
{ };

// template loop unrolling
//
template<class Expr, int N>
class tVBoolSplitLoop<Expr, N, true>
{
   public:
      typedef bool value_type;

      inline static bool anyof( Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNBoolLoop<Expr, N-1>::anyof(e);         
      }
      inline static bool allof( Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNBoolLoop<Expr, N-1>::allof(e);         
      }
      inline static bool noneof( Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tNBoolLoop<Expr, N-1>::noneof(e);         
      }

};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class Expr, int N>
class tVBoolSplitLoop<Expr, N, false>
{
   public:
      typedef bool value_type;

      inline static bool anyof( Expr& e )
      {
         for(int i=0; i<N; ++i) 
            if( e[i] ) return true;
         return false;
      }
      inline static bool allof( Expr& e )
      {
         for(int i=0; i<N; ++i) 
            if( !e[i] ) return false;
         return true;
      }
      inline static bool noneof( Expr& e )
      {
         for(int i=0; i<N; ++i) 
            if( e[i] ) return false;
         return true;
      }
};


// ==================================================================

//
// template loop
//
template<class Expr, int N>
class tNBoolLoop
{
   public:
      typedef bool value_type;

      inline static bool anyof( Expr& e )
      {
         if( e[N] ) return true;
         else return tNBoolLoop<Expr, N-1>::anyof(e);
      }
      inline static bool allof( Expr& e )
      {
         if( !e[N] ) return false;
         else return tNBoolLoop<Expr, N-1>::allof(e);
      }
      inline static bool noneof( Expr& e )
      {
         if( e[N] ) return false;
         else return tNBoolLoop<Expr, N-1>::noneof(e);
      }
};


// end of recursion
template<class Expr>
class tNBoolLoop<Expr, 0>
{
   public:
      typedef bool value_type;

      inline static bool anyof( Expr& e )
      {
         if( !e[0] ) return false;
         else return true;
      }
      inline static bool allof( Expr& e )
      {
         if( !e[0] ) return false;
         else return true;
      }
      inline static bool noneof( Expr& e )
      {
         if( e[0] ) return false;
         else return true;
      }
};

}

#endif //__LTL_FBOOL__
