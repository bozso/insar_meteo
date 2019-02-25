/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testuserfuncs.cpp 476 2010-11-12 06:00:58Z drory $
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
#define LTL_RANGE_CHECKING
#define LTL_USE_SIMD
#define LTL_DEBUG_EXPRESSIONS

#include <ltl/marray.h>
#include <ltl/statistics.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

float user_func( float x )
{
   return x*x;
}
VEC_TYPE(float) user_func_vec( VEC_TYPE(float) x )
{
     return __ltl_TMul<float,float>::eval_vec(x,x);
}

float user_func_novec( float x )
{
   return x*x;
}

DECLARE_UNARY_FUNC( user_func, float )
DECLARE_UNARY_FUNC_VEC( user_func, float, user_func_vec );

DECLARE_UNARY_FUNC( user_func_novec, float )

struct functor
{
   typedef float value_type;
   enum { isVectorizable = 1 };

   float operator()( float a )
   {
      return a*a;
   }

   VEC_TYPE(float) operator()( VEC_TYPE(float) a )
   {
      return __ltl_TMul<float,float>::eval_vec(a,a);
   }
};

int main(int argc, char **argv)
{
   cout << "Testing user supplied functions ... " << endl;

   MArray<float,1> A(100), B(100);
   A = indexPosInt(A,1);

   cout << "Should be vectorized :" << endl;
   B = user_func(A);
   LTL_ASSERT_( allof(B==A*A), "User supplied vectorized function failed" );

   cout << "Should NOT be vectorized :" << endl;
   B = user_func_novec(A);
   LTL_ASSERT_( allof(B==A*A), "User supplied vectorized function failed" );

   cout << "Testing apply() with user supplied functor ... " << endl;

   cout << "Should be vectorized :" << endl;
   functor f;
   B = apply(f,A);
   LTL_ASSERT_( allof(B==A*A), "User supplied vectorized functor apply() failed" );
}
