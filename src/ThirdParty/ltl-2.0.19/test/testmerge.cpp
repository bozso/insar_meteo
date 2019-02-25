/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmerge.cpp 476 2010-11-12 06:00:58Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C) Niv Drory <drory@mpe.mpg.de>
*               Claus A. Goessl <cag@usm.uni-muenchen.de>
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
//#define LTL_USE_SIMD

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

struct functor
{
   typedef float value_type;
   enum { isVectorizable = 0 };

   value_type operator()( float a )
   {
      return a*a;
   }

};

struct functor2
{
   typedef float value_type;
   enum { isVectorizable = 0 };

   value_type operator()( float a, float b )
   {
      return a*b;
   }

};


int main(int argc, char **argv)
{
   cerr << "Testing MArray where(), merge(), and apply() ..." << endl;

   MArray<float,2> A(5,5);
   A = 0.0f;
   A.setBase(-1,-1);

   A(0,0)  = 1.f;
   A(-1,0) = 3.f;
   MArray<float,2>::IndexSet l = where( A!=0.0f );
   LTL_ASSERT_( l.size() == 2,
                "MArray where() failed" );

   MArray<float,2>::IndexSet l2 = where( (A+A-A) > 0.0f );
   LTL_ASSERT_( l2.size() == 2,
                "MArray where() failed" );

   MArray<float,2>::IndexSet l3 = where( (A+A-A) == 0.0f );
   LTL_ASSERT_( l3.size() == (A.nelements()-l2.size()),
                "MArray where() failed" );

   A[l] = 5.0f;
   LTL_ASSERT_( A(0,0) == 5.0f && A(-1,0) == 5.0f,
                "MArray where()/operator[] failed" );
   A[l] = 0.0f;
   LTL_ASSERT_( noneof(A),
                "MArray where()/operator[] failed" );


   A(-1,Range::all()) = 2.0f;
   A = merge( A, 1.0f/A, -1.0f );
   LTL_ASSERT_( allof( A(-1,Range::all()) == 0.5f ),
                "MArray merge() failed" );

   A(-1,Range::all()) = -1.0f;
   LTL_ASSERT_( allof( A == -1.0f ),
                "MArray merge() failed" );


   functor F;
   MArray<float,1> B(10), C(10), D(10);
   B = 1,2,3,4,5,6,7,8,9,10;
   C = 0;

   C = apply( F, B );
   LTL_ASSERT_( allof(C==B*B),
                "MArray apply() failed" );

   C = apply( F, B*B );
   LTL_ASSERT_( allof(C==B*B*B*B),
                "Expression apply() failed" );


   functor2 F2;
   D = 0;
   D = apply( F2, B, B );
   LTL_ASSERT_( allof(D==B*B),
                "MArray binary apply failed");
   D = apply( F2, B, 2*B );
   LTL_ASSERT_( allof(D==B*2*B),
                "MArray,Expr binary apply failed");
   D = apply( F2, 2*B, B );
   LTL_ASSERT_( allof(D==2*B*B),
                "Expr,MArray binary apply failed");
   D = apply( F2, 2*B, 3*B );
   LTL_ASSERT_( allof(D==6*B*B),
                "Expr,Expr binary apply failed");
}

