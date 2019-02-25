/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmarray.cpp 322 2008-02-19 16:21:09Z drory $
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
#include <ltl/marray/blas.h>
#include <ltl/statistics.h>
#include <ltl/marray_io.h>
#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

void test_blas1( void );
void test_blas2( void );
void test_blas3( void );
void test_blas4( void );

int main(int argc, char **argv)
{
   cerr << "Testing MArray-BLAS interface ..." << endl;
   test_blas1();
   test_blas2();
   test_blas3();
   test_blas4();
}

void test_blas1( void )
{
   cerr << "Testing BLAS level 1 funtions:" << endl;

   MArray<double,1> x(10), y(10);
   x = 1,2,3,4,5,6,7,8,9,10;
   y = 1,2,3,4,5,6,7,8,9,10;

//   float xx[] = {1,2,3,4,5,6,7,8,9,10};
//   float yy[] = {1,2,3,4,5,6,7,8,9,10};
//   int nn = 10, incx=1;
//   cout << sdot_(&nn, xx, &incx, yy, &incx) << endl;

   double a = blas_dot( x, y );
   double b = sum(x*y);
   cout << a << " " << b << endl;
   LTL_ASSERT_( fabs(a-b)<1e-10, "BLAS dot() failed.");

   double alpha = 0.5;
   x = 2;
   y = 3;
   blas_axpy(x, alpha, y);
   cout << y << endl;
   LTL_ASSERT_( allof(y==4), "BLAS saxpy() failed." );

   MArray<double,1> xd(1001), yd(1001);
   double alphad = 1.5;

   xd = 8;
   yd = 4;

   blas_axpy(xd, alphad, yd);
   //out << yd << endl;
   LTL_ASSERT_( allof(yd==16), "BLAS daxpy() failed." );
}

void test_blas2( void )
{
   cerr << "Testing BLAS level 2 funtions:" << endl;

   MArray<float,1> x(4), x2(3), y1(3), y2(4);
   x = 1,2,3,4;
   x2 = 1,2,3;
   y1 = 30, 70, 110;
   y2 = 38, 44, 50, 56;

   MArray<float,2> A(3,4);
   A = 1,5,9,
       2,6,10,
       3,7,11,
       4,8,12;  // column major order ...

   MArray<float,1> yy = blas_gemv( A, x );
   cout << yy << endl;
   LTL_ASSERT_( allof(yy==y1), "BLAS sgemv() failed." );

   MArray<float,1> yyy = blas_gemv( A, x2, true );
   cout << yyy << endl;
   LTL_ASSERT_( allof(yyy==y2), "BLAS sgemv() transposed failed." );
}

void test_blas3( void )
{
   cerr << "Testing BLAS level 3 funtions:" << endl;

   MArray<double,2> A(2,3);
   A = 1, -1,
       0, 3,
       2, 1;  // column major order ...
   MArray<double,2> B(3,2);
   B = 3, 2, 1,
       1, 1, 0; // column major order ...
   MArray<double,2> C(2,2), CC(2,2);
   C = 5, 4,
       1, 2;
   CC = 0.0;

   blas_gemm( 1.0, A, B, 0.0, CC );
   cout << CC << endl;
   LTL_ASSERT_( allof(CC==C), "BLAS dgemm() failed." );

   MArray<double,2> D(3,3), DD(3,3);
   D = 2, 3, 7,
       1, 3, 5,
       1, 0, 2;
   DD = 0.0;
   blas_gemm( 1.0, A, B, 0.0, DD, true, true );
   cout << DD << endl;
   LTL_ASSERT_( allof(DD==D), "BLAS dgemm() transposed failed." );
}

void test_blas4( void )
{
   cerr << "Testing BLAS dot with subarrays:" << endl;

   MArray<float,2> G(5,5);
   G = 2, 3, 7, 3, 1,
       1, 3, 5, 2, 1,
       1, 2, 2, 1, 3,
       2, 3, 1, 4, 2,
       1, 5, 2, 3, 4;

   const int npar = G.length(1);
   for (int i=1; i<=npar; ++i)
      for (int j=1; j<=npar; ++j)
      {
         double tmp1 = blas_dot (G(i,Range::all()), G(Range::all(),j));
         double tmp2 = sum(G(i,Range::all()) * G(Range::all(),j) );
         cout << i << "," << j << " : " << tmp1 << "  " << tmp2 << endl;
         LTL_ASSERT_( fabs(tmp1-tmp2)<1e-6, "BLAS ddot() on subarrays failed." );
      }
}
