/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvtloops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_FVTLOOPS__
#define __LTL_FVTLOOPS__

#include <ltl/config.h>

namespace ltl {

/*! \file fvtloops.h
  Template metaprograms - for loop:
  These are used in ltl::FVector and ltl::FMatrix operator= and operator X=
  for performing the loop in elementwise operations.
*/

template<class A, class B, class operation, int N>
class tNLoop;

template<class A, class B, class operation, int N, bool unroll>
class tFVSplitLoop;

/*!
  \code
  for(int i=1; i<=N ; ++i) operation::eval(a[i], b[i]);
  \endcode
  Uses full unrolling for N<=LTL_TEMPLATE_LOOP_LIMIT (config.h)
  and a for-loop otherwise.
*/
template<class A, class B, class operation, int N>
class tFVLoop
{
   public:
      typedef typename operation::value_type value_type;

      enum { static_size = N * (A::static_size + B::static_size) };
   
      // for per element operations which store their result to A
      inline static void eval(A& a, const B& b)
      {
         tFVSplitLoop<A, B, operation, N,
                         (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                         >::eval(a, b);
      }
};


// ==================================================================

//
//! Wrapper to decide whether template loop is ok or simple C-loop does better.
//
template<class A, class B, class operation, int N, bool unroll>
class tFVSplitLoop
{ };

//! Template loop unrolling.
//
template<class A, class B, class operation, int N>
class tFVSplitLoop<A, B, operation, N, true>
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval(A& a, const B& b)
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         tNLoop<A, B, operation, N-1>::eval(a, b);
      }
};


//! We exceed the limit for template loop unrolling: Simple for-loop.
template<class A, class B, class operation, int N>
class tFVSplitLoop<A, B, operation, N, false>
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval(A& a, const B& b)
      {
         for(int i=0; i<N; ++i)
            operation::eval(a[i], b[i]);
      }
};


// ==================================================================

//
//! Template loop: \code for(int i=0; i < N; ++i) operation::eval(a, b);\endcode
//
template<class A, class B, class operation, int N>
class tNLoop
{
   public:
      typedef typename operation::value_type value_type;

      inline static void eval(A& a, const B& b)
      {
         tNLoop<A, B, operation, N-1 >::eval(a, b);
         operation::eval( a[N], b[N] );
      }
};


//! End of recursion.
template<class A, class B, class operation>
class tNLoop<A, B, operation, 0>
{
   public:
      inline static void eval(A& a, const B& b)
      {
         operation::eval( a[0], b[0] );
      }
};


// ==================================================================
// ==================================================================


//
// template metaprogram - for loop:
// this is used in FVector and FMatrix operator= and operator X=
// for performing the loop in elementwise operations
//

template<class A, class B, int N>
class tNSwap;

template<class A, class B, int N, bool unroll>
class tFVSplitSwap;

//
// for(int i=1; i<=N ; ++i) {dummy = a[i]; a=[i] = b[i]; b[i] = dummy;}
// uses full unrolling for N<=LTL_TEMPLATE_LOOP_LIMIT (config.h)
// and a for-loop otherwise
//
template<class A, class B, int N>
class tFVSwap
{
   public:
      enum { static_size = N * (A::static_size + B::static_size)};
   
      // for per element operations which store their result to A
      inline static void eval(A& a, B& b)
      {
         tFVSplitSwap<A, B, N,
                      (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
                       >::eval(a, b);
      }
};


// ==================================================================

//
// wrapper to decide whether template loop is ok or
// simple C-loop does better
//

template<class A, class B, int N, bool unroll>
class tFVSplitSwap
{ };

// template loop unrolling
//
template<class A, class B, int N>
class tFVSplitSwap<A, B, N, true>
{
   public:
      inline static void eval(A& a, B& b)
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         tNSwap<A, B, N-1>::eval(a, b);
      }
};


// we exceed the limit for template loop unrolling:
// simple for-loop
//
template<class A, class B, int N>
class tFVSplitSwap<A, B, N, false>
{
   public:
      inline static void eval(A& a, B& b)
      {
         for(int i=0; i<N; ++i)
         {
            const typename A::value_type dummy = a[i];
            a[i] = b[i];
            b[i] = dummy;
         }
      }
};


// ==================================================================

//
// template loop
// for(int i=0; i < N; ++i) operation::eval(a, b)
//
template<class A, class B, int N>
class tNSwap
{
   public:
      inline static void eval(A& a, B& b)
      {
         tNSwap<A, B, N-1 >::eval(a, b);
         const typename A::value_type dummy = a[N];
         a[N] = b[N];
         b[N] = dummy;
      }
};


// end of recursion
template<class A, class B>
class tNSwap<A, B, 0>
{
   public:
      inline static void eval(A& a, B& b)
      {
         const typename A::value_type dummy = a[0];
         a[0] = b[0];
         b[0] = dummy;
      }
};

}

#endif //__LTL_FVTLOOPS__
