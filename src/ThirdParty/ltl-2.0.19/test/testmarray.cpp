/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmarray.cpp 562 2015-04-30 16:01:16Z drory $
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
//#define LTL_USE_SIMD

#include <ltl/marray.h>
#include <ltl/statistics.h>
#include <ltl/marray_io.h>
#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

template<typename T>
void test_logic(void)
{
   MArray<T,1>A(10);
   A = 0;

   LTL_ASSERT_( !allof(A),
                "MArray allof() expression failed" );
   LTL_ASSERT_( !anyof(A),
                "MArray anyof() expression failed" );
   LTL_ASSERT_( noneof(A),
                "MArray noneof() expression failed" );
   LTL_ASSERT_( count(A)==0,
                "MArray count() expression failed" );

   A(3) = 1;
   A(7) = 1;

   LTL_ASSERT_( !allof(A),
                "MArray allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "MArray anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "MArray noneof() expression failed" );
   LTL_ASSERT_( count(A)==2,
                "MArray count() expression failed" );

   A = 7;
   LTL_ASSERT_( allof(A),
                "MArray allof() expression failed" );
   LTL_ASSERT_( anyof(A),
                "MArray anyof() expression failed" );
   LTL_ASSERT_( !noneof(A),
                "MArray noneof() expression failed" );
   LTL_ASSERT_( count(A)==10,
                "MArray count() expression failed" );
}

template<typename T>
void test_init(void)
{
   MArray<T,1> A(5);
   for( int i=1; i<=(int)A.nelements(); ++i)
      A(i) = T(i);

   MArray<T,1> B(5);
   B = 1,2,3,4,5;

   LTL_ASSERT( allof(A==B), "Basic list initializer failed");
}

template<typename T>
void test_init2(void)
{
   MArray<T,1> A(5);
   A = 1,2,3,4,5;

   MArray<T,1> B;
   B = A;
   LTL_ASSERT( allof(B==A), "Assignment to uninitialized MArray failed");

   MArray<T,1> C;
   C = A+B-A;
   LTL_ASSERT( allof(C==B), "Assignment of expression to uninitialized MArray failed");
}

template<typename T>
void test_init3(void)
{
   MArray<T,1> A(5);
   for( int i=1; i<=(int)A.nelements(); ++i)
      A(i) = T(i);

   MArray<T,1> B(5);
   B = A(1),A(2),A(3),A(4),A(5);
   LTL_ASSERT( allof(A==B), "Advanced list initializer 1 failed");

   MArray<T,1> C(5);
   C = sin(M_PI)+A(1)+5,A(2)+4,A(3)+B(3),A(4)+B(2),A(5)+2-A(1)+sin(M_PI);
   LTL_ASSERT( allof(C==6), "Advanced list initializer 2 failed");
}

void test_copy(void)
{
   MArray<float,1> A, B;
   A=B;
   LTL_ASSERT( A.empty() && B.empty(), "Assingment of empty MArray failed" );

   A.makeReference(MArray<float,1>(10));
   A=B;
   LTL_ASSERT( A.empty() && B.empty(), "Assingment of empty MArray to non empty MArray failed" );

   B.makeReference(MArray<float,1>(10));
   A=B;
   LTL_ASSERT( !A.empty() && !B.empty(), "Assingment of non empty MArray to empty MArray failed" );

   MArray<float,1> C;
   MArray<float,1> D(C);
   LTL_ASSERT( C.empty() && D.empty(), "Copy-ctor with empty MArray failed" );
}

void test_expr1(void)
{
   MArray<int,1> A(10);
   MArray<float,1> B(10);
   char c =1;
   short s=1;
   int i=1;
   long l=1;
   float f=1.1f;
   double d=1.1;
   A = 1;
   B = A + c;
   LTL_ASSERT( allof(B==2.0f), "Expression involving foreign literal type failed.");
   B = 0.0f;
   B = s + A;
   LTL_ASSERT( allof(B==2.0f), "Expression involving foreign literal type failed.");
   B = 0.0f;
   B = A + i;
   LTL_ASSERT( allof(B==2.0f), "Expression involving foreign literal type failed.");
   B = 0.0f;
   B = l + A;
   LTL_ASSERT( allof(B==2.0f), "Expression involving foreign literal type failed.");
   B = 0.0f;
   B = A + f;
   LTL_ASSERT( allof(B==2.1f), "Expression involving foreign literal type failed.");
   B = 0.0f;
   B = A + d;
   LTL_ASSERT( allof(B==2.1f), "Expression involving foreign literal type failed.");
}

void test_expr2(void)
{
   MArray<float,1> D(10);
   D = 1.5f;
   D = D + (D - 1) + 1 - D;
   LTL_ASSERT_( allof(D==1.5f), "Large expression involving foreign literal type failed.");

   short s = 1;
   D = D + (D - s) + 1.0 - D;
   LTL_ASSERT_( allof(D==1.5f), "Large expression involving foreign literal type failed.");
}



void test_referenceN(void)
{
   MArray<float,2> K(10,10);
   K = 10.0f;
   MArray<float,1> M;
   int dims[1];
   dims[0] = 100;

   M.makeReferenceWithDims(K,dims);

   LTL_ASSERT_( allof(M==K(1,1)), "makeReferenceWithDims failed.");
}




int main(int argc, char **argv)
{
   cerr << "Testing MArray initialization ..." << endl;

   MArray<float,2> A(5,5);
   A = 42.f;

   for( int i=1; i<=5; ++i )
      for( int j=1; j<=5; ++j )
         LTL_ASSERT_( A(i,j)==42.f, "MArray init failed" );

   cerr << "Testing copy assignment"  << endl;
   test_copy();

   cerr << "Testing MArray anyof/allof/noneof/count logic ..." << endl;
   test_logic<char>();
   test_logic<short>();
   test_logic<int>();
   test_logic<float>();
   test_logic<double>();

   cerr << "Testing MArray list initializer" << endl;
   test_init<char>();
   test_init<int>();
   test_init<float>();
   test_init<double>();

   cerr << "Testing MArray basics ..." << endl;
   A.setBase(-1,-1);
   LTL_ASSERT_( A(-1,-1)==42.f && A(1,1)==42.f,
                "MArray setBase() failed" );
   A *= 2.0f;
   A -= A/2.f;
   cout << A << endl;
   LTL_ASSERT_( allof(A==42.f),
                "MArray basic expressions failed" );

   LTL_ASSERT_( allof(feq(A,42.f)), "MArray feq failed");
   LTL_ASSERT_( allof(fneq(A,43.f)), "MArray fneq failed");

   cerr << "Testing literals in expressions ..." << endl;
   test_expr1();
   test_expr2();

   cerr << "Testing MArray basic features  ..." << endl;

   MArray<int, 2> I(5,5), I2(5,5);
   I = cast<int>()(A);
   I2 = 42;

   LTL_ASSERT_( allof(I == I2),
                "MArray type-cast failed" );

   I = cast<int>()(2*A-42.0f);
   I2 = 42;

   LTL_ASSERT_( allof(I == I2),
                "MArray expression type-cast failed" );

   MArray<float,1> B = A(-1,Range::all());
   B = 1.f;
   LTL_ASSERT_( allof(B==1.f) && allof( A(-1,Range::all())==1.f ),
                "MArray slice/allof() 1 failed" );

   MArray<float,1> C = A(Range::all(),1);
   C = 2.f;
   LTL_ASSERT_( allof(C==2.f) && allof( A(Range::all(),1)==2.f ),
                "MArray slice/allof() 2 failed" );

   A = 10.0f;
   A = log10(A);
   LTL_ASSERT_( allof(A==1.0f),
                "MArray basic mathlib expressions failed" );

   MArray<float,1> A1(10);
   A1 = indexPosFlt(A1,1);
   LTL_ASSERT_( A1(1)==1.0f && A1(2)==2.0f && A1(9)==9.0f && A1(10)==10.0f ,
                "MArray 1-D indexPos() failed" );

   MArray<float,3> AA(10,10,10);
   AA = indexPosFlt(AA,1) + 10*indexPosFlt(AA,2) + 100*indexPosFlt(AA,3);
   LTL_ASSERT_( AA(1,1,1)==111.0f && AA(6,9,2)==296.0f && AA(9,9,9)==999.0f ,
                "MArray 3-D indexPos() failed" );

   MArray<float,2> BB = AA(Range(2,7,2),5,Range(3,8,3));
   MArray<float,2> CC(3,2);
   CC = 352.f,354.f,356.f,
        652.f,654.f,656.f;
   LTL_ASSERT_( allof(BB==CC),
                "MArray subarray failed" );

   MArray<float,1> DD = BB(2,Range::all());
   MArray<float,1> EE(2);
   EE = 354.f,654.f;
   LTL_ASSERT_( allof(DD==EE),
                "MArray complex slice/allof() failed" );

   DD = 555.f;
   LTL_ASSERT_( allof(DD==555.f),
                "MArray complex slice assign failed" );
   LTL_ASSERT_( AA(4,5,3)==555.f && AA(4,5,6)==555.f,
                "MArray complex slice assign failed" );

   cerr << "Testing Assignment to uninitialized arrays" << endl;
   test_init2<int>();
   test_init2<float>();

   cerr << "Testing List initialization" << endl;
   test_init3<int>();
   test_init3<float>();

   cerr << "Testing references with changing rank" << endl;
   test_referenceN();

}

