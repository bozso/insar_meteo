/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testpartialreduc.cpp 476 2010-11-12 06:00:58Z drory $
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

#include <ltl/marray.h>
#include <ltl/statistics.h>
#include <ltl/marray_io.h>
#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

template <typename T>
void test1(void)
{
   MArray<T,2> A(5,4);
   A = indexPosInt(A,1) + 10*indexPosInt(A,2);
   //cout << A << endl;

   MArray<T,1> B(4), C(5);
   MArray<T,1> BB(4), CC(5);
   BB = 65, 115, 165, 215;
   CC = 104, 108, 112, 116, 120;

   B = partial_sum( A, 1 );
   cout << B << endl;
   LTL_ASSERT_( allof(B==BB), "Partial reduction sum failed");

   C = partial_sum( A, 2 );
   cout << C << endl;
   LTL_ASSERT_( allof(C==CC), "Partial reduction sum failed");
}

template <typename T>
void test2(void)
{
   MArray<T,2> A1(5,4), A2(5,4);
   A1 = indexPosInt(A1,1);
   A2 = 10*indexPosInt(A2,2);

   MArray<T,1> B(4), C(5);
   MArray<T,1> BB(4), CC(5);
   BB = 65, 115, 165, 215;
   CC = 104, 108, 112, 116, 120;

   B = partial_sum( A1+A2, 1 );
   cout << B << endl;
   LTL_ASSERT_( allof(B==BB), "Partial reduction sum failed");

   C = partial_sum( A1+A2, 2 );
   cout << C << endl;
   LTL_ASSERT_( allof(C==CC), "Partial reduction sum failed");
}

template <typename T>
void test3(void)
{
   MArray<T,2> A(5,4);
   A = indexPosInt(A,1) + 10*indexPosInt(A,2);
   A(1,1) = 0;
   A(2,1) = 0;
   A(1,4) = 0;
   //cout << A << endl;

   MArray<T,1> B(4), C(5);
   MArray<T,1> BB(4), CC(5);
   BB = 0, 1, 1, 0;
   CC = 0, 0, 1, 1, 1;

   B = partial_allof( A, 1 );
   cout << B << endl;
   LTL_ASSERT_( allof(B==BB), "Partial reduction allof failed");

   C = partial_allof( A, 2 );
   cout << C << endl;
   LTL_ASSERT_( allof(C==CC), "Partial reduction allof failed");
}

template <typename T>
void test4(void)
{
   MArray<T,3> A(5,4,3);
   A = indexPosInt(A,1) + 10*indexPosInt(A,2) + 100*indexPosInt(A,3);
   //cout << A << endl;

   MArray<T,2> B(5,3);
   MArray<T,2> BB(5,3);
   BB = 0;
   for( int i=1; i<=5; ++i)
      for( int j=1; j<=3; ++j)
         for( int k=1; k<=4; ++k)
            BB(i,j) += A(i,k,j);
   B = partial_sum( A, 2 );
   cout << B << endl;
   LTL_ASSERT_( allof(B==BB), "Partial reduction along middle dimension failed");
}

int main(int argc, char **argv)
{
   cerr << "Testing partial reductions ..." << endl;

   test1<double>();
   test1<float>();
   test1<int>();
   test1<short>();
   test2<float>();
   test2<double>();
   test2<int>();
   test2<short>();
   test3<int>();
   test3<float>();
   test4<float>();
}

