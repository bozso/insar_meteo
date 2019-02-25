/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmarrayio.cpp 476 2010-11-12 06:00:58Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C)  Jan Snigula <snigula@usm.uni-muenchen.de>
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

#include <fstream>

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/fvector.h>
#include <ltl/fmatrix.h>
#include <ltl/statistics.h>

using namespace ltl;
using namespace std;

using std::cout;
using std::endl;

void test_marray_io1();
void test_marray_io1_expr();
void test_marray_io2();
void test_marray_io2_expr();
void test_marray_io3();
void test_fvector_io();
void test_fmatrix_io();

int main(int argc, char **argv)
{
   test_marray_io1();
   test_marray_io1_expr();
   test_marray_io2();
   test_marray_io2_expr();
   test_marray_io3();
   test_fvector_io();
   test_fmatrix_io();
}

void test_marray_io1(void)
{
   MArray<float,1> A(10);
   MArray<float,1> B(10);
   A = 0,1,2,3,4,5,6,7,8,9;
   B = 0;
   ofstream os( "testmarrayio0.dat" );
   os << A << endl;
   os.close();
   
   ifstream is( "testmarrayio0.dat" );
   is >> B;
   is.close();
   LTL_ASSERT_( allof(A == B),
                "MArray stream io failed" );
}

void test_marray_io1_expr(void)
{
   MArray<float,1> A(10);
   MArray<float,1> B(10);
   A = 0,1,2,3,4,5,6,7,8,9;
   B = 0;
   ofstream os( "testmarrayio0.dat" );
   os << (A+1.0f) << endl;
   os.close();

   ifstream is( "testmarrayio0.dat" );
   is >> B;
   is.close();
   LTL_ASSERT_( allof((A+1.0f) == B),
                "MArray stream io failed" );
}

void test_marray_io2(void)
{
   MArray<float,2> A(10,10);
   MArray<float,2> B(10,10);
   A = indexPosInt(A,1)+indexPosInt(A,2)*10;
   B = 0;
   ofstream os( "testmarrayio0.dat" );
   os << A << endl;
   os.close();

   ifstream is( "testmarrayio0.dat" );
   is >> B;
   is.close();
   LTL_ASSERT_( allof(A == B),
                "MArray stream io failed" );
}

void test_marray_io2_expr(void)
{
   MArray<float,2> A(10,10);
   MArray<float,2> B(10,10);
   ofstream os( "testmarrayio0.dat" );
   os << indexPosInt(A,1)+indexPosInt(A,2)*10 << endl;
   os.close();

   A = indexPosInt(A,1)+indexPosInt(A,2)*10;
   B = 0;
   ifstream is( "testmarrayio0.dat" );
   is >> B;
   is.close();
   LTL_ASSERT_( allof(A == B),
                "MArray stream io failed" );
}

void test_marray_io3(void)
{
   MArray<float,3> A(10,10,10);
   MArray<float,3> B(10,10,10);
   A = indexPosInt(A,1)+indexPosInt(A,2)*10+indexPosInt(A,3)*100;
   B = 0;
   ofstream os( "testmarrayio0.dat" );
   os << A << endl;
   os.close();

   ifstream is( "testmarrayio0.dat" );
   is >> B;
   is.close();
   LTL_ASSERT_( allof(A == B),
                "MArray stream io failed" );
}

void test_fvector_io(void)
{
   FVector<float,3> X;
   FVector<float,3> Y;
   X = 0,1,2;
   Y = 0;
   ofstream os2( "testmarrayio1.dat" );
   os2 << X << endl;
   os2.close();
   
   ifstream is2( "testmarrayio1.dat" );
   is2 >> Y;
   LTL_ASSERT_( X(1) == Y(1) && X(2) == Y(2) && X(3) == Y(3),
                "FVector stream io failed" );
}


void test_fmatrix_io(void)
{
   FMatrix<float,3,4> X;
   FMatrix<float,3,4> Y;
   X = 0,1,2,3,4,5,6,7,8,9,10,11;
   Y = 0;
   ofstream os2( "testmarrayio2.dat" );
   os2 << X << endl;
   os2.close();

   ifstream is2( "testmarrayio2.dat" );
   is2 >> Y;
   LTL_ASSERT_( allof(X==Y),
                "FMatrix stream io failed" );
}
