/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmtloops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FMTLOOPS__
#define __LTL_FMTLOOPS__

#include <ltl/config.h>

namespace ltl {

/*! \file fmtloops.h
  Template metaprograms - for loop:
  These are used in ltl::FMatrix::operator= and operator X=
  for performing the loop in elementwise operations.
*/

template<class A, class B, class operation, int N, int I, int J>
class tMNLoop;

template<class A, class B, class operation, int M, int N, bool unroll>
class tFMSplitLoop;

//
// for( int i=0; i<M ; ++i ) 
//   for( int i=0; i<N; ++i )
//     operation::eval(a(i,j), b(i,j))
// uses full unrolling for M*N<=LTL_TEMPLATE_LOOP_LIMIT (config.h)
// and a for-loop otherwise
//
template<class A, class B, class operation, int M, int N>
class tFMLoop
{
   public:
      typedef typename operation::value_type value_type;

      enum { static_size = A::static_size + B::static_size };
   
      // for per element operations which store their result to A
      inline static void eval( A& a, const B& b )
      {
         tFMSplitLoop<A, B, operation, M, N,
                     (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                     >::eval( a, b );
      }
};


// ==================================================================

//
// wrapper to decide whether template loop is ok or
// simple C-loop does better
//

template<class A, class B, class operation, int M, int N, bool unroll>
class tFMSplitLoop
{ };

// template loop unrolling
//
template<class A, class B, class operation, int M, int N>
class tFMSplitLoop<A, B, operation, M, N, true>
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval( A& a, const B& b )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         tMNLoop<A, B, operation, N, M, N>::eval( a, b );
      }
};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class A, class B, class operation, int M, int N>
class tFMSplitLoop<A, B, operation, M, N, false>
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval( A& a, const B& b )
      {
         for( int i=1; i<=M; ++i )
            for( int j=1; j<=N; ++j )
               operation::eval( a(i,j), b(i,j) );
      }
};


// ==================================================================

//
// template loop
//
template<class A, class B, class operation, int N, int I, int J>
class tMNLoop
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval(A& a, const B& b)
      {
         tMNLoop<A, B, operation, N, I, J-1>::eval( a, b );
         operation::eval( a(I,J), b(I,J) );
      }
};


// end of row
template<class A, class B, class operation, int N, int I>
class tMNLoop<A, B, operation, N, I, 1>
{
   public:
      inline static void eval(A& a, const B& b)
      {
         tMNLoop<A, B, operation, N, I-1, N>::eval( a, b );
         operation::eval( a(I,1), b(I,1) );
      }
};


// end of recursion
template<class A, class B, class operation, int N>
class tMNLoop<A, B, operation, N, 1, 1>
{
   public:
      inline static void eval(A& a, const B& b)
      {
         operation::eval( a(1,1), b(1,1) );
      }
};

}

#endif //__LTL_FMTLOOPS__
