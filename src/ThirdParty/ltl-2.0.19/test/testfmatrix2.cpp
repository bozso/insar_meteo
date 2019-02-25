/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testfmatrix2.cpp 562 2015-04-30 16:01:16Z drory $
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
//#define LTL_DEBUG_EXPRESSIONS

#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/fmatrix.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

template<typename T>
void test_logic(void)
{
   FMatrix<T,3,3>A;
   A = 0;

   LTL_ASSERT_( !allof(A),
                "FMatrix allof() expression failed" );
   LTL_ASSERT_( !anyof(A),
                "FMatrix anyof() expression failed" );
   LTL_ASSERT_( noneof(A),
                "FMatrix noneof() expression failed" );

   A(1,3) = 1;
   A(2,1) = 1;

   LTL_ASSERT_( !allof(A),
                "FMatrix allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "FMatrix anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "FMatrix noneof() expression failed" );

   A = 7;
   LTL_ASSERT_( allof(A),
                "FMatrix allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "FMatrix anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "FMatrix noneof() expression failed" );
}

template<typename T>
void test_init(void)
{
   FMatrix<T,3,3> A;
   for( int i=1; i<=3; ++i)
      for( int j=1; j<=3; ++j)
      A(i,j) = T(i);

   FMatrix<T,3,3> B;
   B = 1,1,1,2,2,2,3,3,3;

   LTL_ASSERT( allof(A==B), "Basic list initializer failed");
}

template<typename T>
void test_init2(void)
{
   FMatrix<T,3,3> A;
   A = 1,2,3,4,5,6,7,8,9;

   FMatrix<T,3,3> B;
   B = A;
   LTL_ASSERT( allof(B==A), "Assignment to uninitialized MArray failed");

   FMatrix<T,3,3> C;
   C = A+B-A;
   LTL_ASSERT( allof(C==B), "Assignment of expression to uninitialized MArray failed");
}

template<typename T>
void test_init3(void)
{
   FMatrix<T,3,3> A;
   for( int i=1; i<=3; ++i)
      for( int j=1; j<=3; ++j)
      A(i,j) = T(i);

   FMatrix<T,3,3> B;
   B = A(1,1),A(1,2),A(1,3),A(2,1),A(2,2),A(2,3),A(3,1),A(3,2),A(3,3);
   LTL_ASSERT( allof(A==B), "Advanced list initializer 1 failed");

   FMatrix<T,3,3> C;
   C = A(1,1)+5,5+A(1,3),A(2,1)+4,2+4,A(3,3)+3,A(2,3)+4,6,6,A(3,2)+3;
   LTL_ASSERT( allof(C==6), "Advanced list initializer 2 failed");
}



int main(int argc, char **argv)
{
   cerr << "Testing FMatrix (for loops) ..." << endl;
   
   FMatrix<float,3,3> A;
   A = 1,2,3,
       4,5,6,
       7,8,9;

   LTL_ASSERT_( (A(1,1) == A[0] && 
                 A(1,1)==1. && A(1,2)==2. && A(1,3)==3. &&
                 A(2,1)==4. && A(2,2)==5. && A(2,3)==6. &&
                 A(3,1)==7. && A(3,2)==8. && A(3,3)==9.),
                "FMatrix basic indexing failed" );

   FMatrix<float,3,3> B = A;
   LTL_ASSERT_( (B(1,1)==1. && B(1,2)==2. && B(1,3)==3. &&
                 B(2,1)==4. && B(2,2)==5. && B(2,3)==6. &&
                 B(3,1)==7. && B(3,2)==8. && B(3,3)==9.),
                "FMatrix init assignment failed" );

   B = 1.;
   LTL_ASSERT_( (B(1,1)==1. && B(1,2)==1. && B(1,3)==1. &&
                 B(2,1)==1. && B(2,2)==1. && B(2,3)==1. &&
                 B(3,1)==1. && B(3,2)==1. && B(3,3)==1.),
                "FMatrix literal assignment failed" );
   
   B = A;
   LTL_ASSERT_( (B(1,1)==1. && B(1,2)==2. && B(1,3)==3. &&
                 B(2,1)==4. && B(2,2)==5. && B(2,3)==6. &&
                 B(3,1)==7. && B(3,2)==8. && B(3,3)==9.),
                "FMatrix assignment failed" );
   
   B = transpose(A);
   LTL_ASSERT_( (B(1,1)==1. && B(1,2)==4. && B(1,3)==7. &&
                 B(2,1)==2. && B(2,2)==5. && B(2,3)==8. &&
                 B(3,1)==3. && B(3,2)==6. && B(3,3)==9.),
                "FMatrix transpose failed" );
   
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

   B = 1.;   
   FMatrix<float,3,3> C = 0.;
   C = A + B + log10(B);
   
   LTL_ASSERT_( (C(1,1)==2. && C(1,2)==3. && C(1,3)==4. &&
                 C(2,1)==5. && C(2,2)==6. && C(2,3)==7. &&
                 C(3,1)==8. && C(3,2)==9. && C(3,3)==10.),
                "FMatrix expression assignment failed" );

   FMatrix<float,3,3> D;
   D = 2,3,4,
       5,6,7,
       8,9,10;

   LTL_ASSERT_( allof( C == D ),
                "FMatrix allof() failed" );
   LTL_ASSERT_( noneof( C != D ),
                "FMatrix noneof() failed" );   
   LTL_ASSERT_( !anyof( C != D ),
                "FMatrix anyof() failed" );

   C *= C;
   
   LTL_ASSERT_( allof( C == pow2(D) ),
                "FMatrix allof(expr) failed" );
   
   FVector<float,3> V1,V2;
   V1 = 1,2,3;
   V2 = 4,5,6;
   FMatrix<float,3,3>::RowVector r = A.row(1);
   LTL_ASSERT_( allof( V1 == r ),
                "FMatrix row() failed" );
   A.swapRows(1, 2);
   LTL_ASSERT_( allof( V2 == r ),
                "FMatrix swaprow() failed" );
   A.swapRows(1, 2);
   LTL_ASSERT_( allof( V1 == r ),
                "FMatrix swaprow() failed" );

   V1 = 1,4,7;
   V2 = 2,5,8;
   FMatrix<float,3,3>::ColumnVector c = A.col(2);
   LTL_ASSERT_( allof( V2 == c ),
                "FMatrix col() failed" );
   A.swapCols(1,2);
   LTL_ASSERT_( allof( V1 == c ),
                "FMatrix swapcol() failed" );
   A.swapCols(1,2);
   LTL_ASSERT_( allof( V2 == c ),
                "FMatrix swapcol() failed" );


   c = V2+1;
   LTL_ASSERT_( (A(1,2)==3. && A(2,2)==6. && A(3,2)==9.),
                "FMatrix column assignment failed" );
}

