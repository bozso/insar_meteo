/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmbool.h 562 2015-04-30 16:01:16Z drory $
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

#ifndef __LTL_FMBOOL__
#define __LTL_FMBOOL__

#include <ltl/config.h>

namespace ltl
{

//
// template metaprogram - bool reductions for FMatrix
//

template<class Expr, int M, int N>              class tMBoolLoop;
template<class Expr, int M, int N, bool unroll> class tMBoolSplitLoop;
template<class Expr, int N, int I, int J>       class tMNBoolLoop;


//---------------------------------------------------------------------
// AT LEAST 1 TRUE
//---------------------------------------------------------------------

template<class Expr, int M, int N>
inline bool anyof( const FMExprNode<Expr,M,N>& e )
{
   return tMBoolLoop< FMExprNode<Expr,M,N>, M,N >::anyof(e);
}

template<class T, int M, int N>
inline bool anyof( FMatrix<T,M,N>& e )
{
   return tMBoolLoop< FMatrix<T,M,N>, M,N >::anyof(e);
}

//---------------------------------------------------------------------
// ALL TRUE
//---------------------------------------------------------------------

template<class Expr, int M, int N>
inline bool allof( const FMExprNode<Expr,M,N>& e )
{
   return tMBoolLoop< FMExprNode<Expr,M,N>, M,N >::allof(e);
}

template<class T, int M, int N>
inline bool allof( FMatrix<T,M,N>& e )
{
   return tMBoolLoop< FMatrix<T,M,N>, M,N >::allof(e);
}


//---------------------------------------------------------------------
// NONE TRUE
//---------------------------------------------------------------------

template<class Expr, int M, int N>
inline bool noneof( const FMExprNode<Expr,M,N>& e )
{
   return tMBoolLoop< FMExprNode<Expr,M,N>, M,N >::noneof(e);
}

template<class T, int M, int N>
inline bool noneof( FMatrix<T,M,N>& e )
{
   return tMBoolLoop< FMatrix<T,M,N>, M,N >::noneof(e);
}


// ======================================================================

template<class Expr, int M, int N>
class tMBoolLoop
{
   public:
      typedef bool value_type;
      enum { static_size = M*N*Expr::static_size };
   
      inline static bool anyof( const Expr& e )
      {
         return tMBoolSplitLoop<Expr, M, N,
                                (Expr::static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::anyof(e);
      }
      inline static bool allof( const Expr& e )
      {
         return tMBoolSplitLoop<Expr, M, N,
                                (Expr::static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::allof(e);
      }
      inline static bool noneof( const Expr& e )
      {
         return tMBoolSplitLoop<Expr, M, N,
                                (Expr::static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                                >::noneof(e);
      }
};



template<class Expr, int M, int N, bool unroll>
class tMBoolSplitLoop
{ };

// template loop unrolling
//
template<class Expr, int M, int N>
class tMBoolSplitLoop<Expr, M, N, true>
{
   public:
      typedef bool value_type;

      inline static bool anyof( const Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tMNBoolLoop<Expr, N, M, N>::anyof(e);         
      }

      inline static bool allof( const Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tMNBoolLoop<Expr, N, M, N>::allof(e);         
      }

      inline static bool noneof( const Expr& e )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         return tMNBoolLoop<Expr, N, M, N>::noneof(e);         
      }
};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class Expr, int M, int N>
class tMBoolSplitLoop<Expr, M, N, false>
{
   public:
      typedef bool value_type;

      inline static bool anyof( const Expr& e )
      {
         for( int i=1; i<=M; ++i )
            for( int j=1; j<=N; ++j ) 
               if( e(i,j) ) return true;
         return false;
      }
      inline static bool allof( const Expr& e )
      {
         for( int i=1; i<=M; ++i )
            for( int j=1; j<=N; ++j ) 
               if( !e(i,j) ) return false;
         return true;
      }
      inline static bool noneof( const Expr& e )
      {
         for( int i=1; i<=M; ++i )
            for( int j=1; j<=N; ++j ) 
               if( e(i,j) ) return false;
         return true;
      }
};


// ==================================================================

//
// template loop
//
//
// template loop
//
template<class Expr, int N, int I, int J>
class tMNBoolLoop
{
   public:
      typedef bool value_type;

      inline static bool anyof( const Expr& e )
      {
         if( e(I,J) ) return true;
         else return tMNBoolLoop<Expr, N, I, J-1>::anyof(e);
      }
      inline static bool allof( const Expr& e )
      {         
         if( !e(I,J) ) return false;
         else return tMNBoolLoop<Expr, N, I, J-1>::allof(e);
      }
      inline static bool noneof( const Expr& e )
      {
         if( e(I,J) ) return false;
         else return tMNBoolLoop<Expr, N, I, J-1>::noneof(e);
      }
};


// end of row
template<class Expr, int N, int I>
class tMNBoolLoop<Expr, N, I, 1>
{
   public:
      typedef bool value_type;

      inline static bool anyof( const Expr& e )
      {
         if( e(I,1) ) return true;
         else return tMNBoolLoop<Expr, N, I-1, N>::anyof(e);
      }
      inline static bool allof( const Expr& e )
      {
         if( !e(I,1) ) return false;
         else return tMNBoolLoop<Expr, N, I-1, N>::allof(e);
      }
      inline static bool noneof( const Expr& e )
      {
         if( e(I,1) ) return false;
         else return tMNBoolLoop<Expr, N, I-1, N>::noneof(e);
      }
};


// end of recursion
template<class Expr, int N>
class tMNBoolLoop<Expr, N, 1, 1>
{
   public:
      typedef bool value_type;

      inline static bool anyof( const Expr& e )
      {
         if( e(1,1) ) return true;
         else return false;
      }
      inline static bool allof( const Expr& e )
      {
         if( !e(1,1) ) return false;
         else return true;
      }
      inline static bool noneof( const Expr& e )
      {
         if( e(1,1) ) return false;
         else return true;
      }
};

}

#endif //__LTL_FMBOOL__
