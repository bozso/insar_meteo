/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testfvec.cpp 562 2015-04-30 16:01:16Z drory $
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

//#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/fvector.h>

#include <iostream>
#include <algorithm>

using namespace ltl;

using std::cout;
using std::endl;

template<typename T>
void test_logic(void)
{
   FVector<T,5>A;
   A = 0;

   LTL_ASSERT_( !allof(A),
                "FVector allof() expression failed" );
   LTL_ASSERT_( !anyof(A),
                "FVector anyof() expression failed" );
   LTL_ASSERT_( noneof(A),
                "FVector noneof() expression failed" );

   A(3) = 1;
   A(5) = 1;

   LTL_ASSERT_( !allof(A),
                "FVector allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "FVector anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "FVector noneof() expression failed" );

   A = 7;
   LTL_ASSERT_( allof(A),
                "FVector allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "FVector anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "FVector noneof() expression failed" );
}

template<typename T>
void test_init(void)
{
   FVector<T,5> A;
   for( int i=1; i<=(int)A.nelements(); ++i)
      A(i) = T(i);

   FVector<T,5> B;
   B = 1,2,3,4,5;

   LTL_ASSERT( allof(A==B), "Basic list initializer failed");
}

template<typename T>
void test_init2(void)
{
   FVector<T,5> A;
   A = 1,2,3,4,5;

   FVector<T,5> B;
   B = A;
   LTL_ASSERT( allof(B==A), "Assignment to uninitialized MArray failed");

   FVector<T,5> C;
   C = A+B-A;
   LTL_ASSERT( allof(C==B), "Assignment of expression to uninitialized MArray failed");
}

template<typename T>
void test_init3(void)
{
   FVector<T,5> A;
   for( int i=1; i<=(int)A.nelements(); ++i)
      A(i) = T(i);

   FVector<T,5> B;
   B = A(1),A(2),A(3),A(4),A(5);
   LTL_ASSERT( allof(A==B), "Advanced list initializer 1 failed");

   FVector<T,5> C;
   C = sin(M_PI)+A(1)+5,A(2)+4,A(3)+B(3),A(4)+B(2),A(5)+2-A(1)+sin(M_PI);
   LTL_ASSERT( allof(C==6), "Advanced list initializer 2 failed");
}


int main(int argc, char **argv)
{
   cerr << "Testing FVector (template unrolled loops) ..." << endl;
   cerr << "Size of FVector<float,3> : " << sizeof( FVector<float,3> ) << ", float[3] : " << sizeof( float[3] ) << endl;

   test_logic<int>();
   test_logic<char>();
   test_logic<float>();
   test_logic<double>();

   test_init<int>();
   test_init<char>();
   test_init<float>();
   test_init<double>();

   test_init2<int>();
   test_init2<char>();
   test_init2<float>();
   test_init2<double>();

   test_init3<int>();
   test_init3<char>();
   test_init3<float>();
   test_init3<double>();

   FVector<float,3> A;
   A = 1,2,3;
   LTL_ASSERT_( (A(1) == A[0] && A(1) == 1. && A(2) == 2. && A(3) == 3.),
                "FVector basic indexing failed" );

   FVector<float,3> B = A;
   LTL_ASSERT_( (B(1) == 1. && B(2) == 2. && B(3) == 3.),
                "FVector init assignment failed" );
   B = 1.;   
   LTL_ASSERT_( (B(1) == 1. && B(2) == 1. && B(3) == 1.),
                "FVector assignment failed" );
   B = 0;
   B = A;
   LTL_ASSERT_( (B(1) == 1. && B(2) == 2. && B(3) == 3.),
                "FVector assignment failed" );
   B -= A;
   LTL_ASSERT_( (B(1) == 0. && B(2) == 0. && B(3) == 0.),
                "FVector assignment failed" );
   B += 1;
   LTL_ASSERT_( (B(1) == 1. && B(2) == 1. && B(3) == 1.),
                "FVector assignment failed" );
   
   FVector<float,3> C = 0.;
   C = A + B + log10(B);
   
   FVector<float,3> D;
   D = 2,3,4;
   LTL_ASSERT_( (C(1) == D(1) && C(2) == D(2) && C(3) == D(3) ),
                "FVector expression failed" );
   LTL_ASSERT_( allof( C == D ),
                "FVector allof() failed" );
   LTL_ASSERT_( noneof( C != D ),
                "FVector noneof() failed" );   
   LTL_ASSERT_( !anyof( C != D ),
                "FVector anyof() failed" );

   D = D + (D - 1) + 1 - D;
   LTL_ASSERT_( (C(1) == D(1) && C(2) == D(2) && C(3) == D(3) ),
                "FVector expression failed" );
   

   C *= C;
   
   LTL_ASSERT_( allof( C == pow2(D) ),
                "FVector allof(expr) failed" );
   
   
   float d = dot(D,D);
   float dd = D(1)*D(1) + D(2)*D(2) + D(3)*D(3);
   LTL_ASSERT_( (d==dd),
                "FVector dot() failed" );

   A = 1,2,3;
   std::copy( A.begin(), A.end(), D.begin() );
   LTL_ASSERT_( (A(1) == D(1) && A(2) == D(2) && A(3) == D(3) ),
                "FVector std::copy() failed" );

}

