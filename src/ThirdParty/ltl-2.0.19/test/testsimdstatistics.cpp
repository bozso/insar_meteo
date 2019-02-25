/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testsimdstatistics.cpp 476 2010-11-12 06:00:58Z drory $
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
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>


using namespace ltl;

using std::cout;
using std::endl;

void test_align1();
void test_align2();

void test_avg();
void test_variance();
template<typename T> void test_minmax(const string& type);


int main(int argc, char **argv)
{
   cerr << "Testing MArray statistics using SIMD vectorization" << endl;

   MArray<float,1> A(100);
   A = indexPosInt(A,1);

   MArray<double,1> D(100);
   D = indexPosInt(D,1);

   MArray<int,1> I(100);
   I = indexPosInt(I,1);
   MArray<float,2> B(10,10);
   B = indexPosInt(B,1) + indexPosInt(B,2)*10;
   float s;

   s = sum(A);
   LTL_EXPECT_( 100*101/2, s, "Vectorized sum() failed" );

   s = sum( B(Range::all(), 5) );
   LTL_EXPECT_( 555, s, "Vectorized sum() failed" );

   s = sum( B(5,Range::all()) );
   LTL_EXPECT_( 600, s, "Non-vectorized sum() failed" );

   A = 1.0f;
   A(45) = 0.0f;
   A(17) = 0.0f;
   s = sum(A,0.0f);
   LTL_EXPECT_( 98, s, "sum<float>() with nan failed" );

   D = 1.0f;
   D(45) = 0.0;
   D(17) = 0.0;
   double t = sum(D,0.0);
   LTL_EXPECT_( 98, t, "sum<double>() with nan failed" );

   I = 1.0f;
   I(45) = 0;
   I(17) = 0;
   int i = sum(I,0);
   LTL_EXPECT_( 98, i, "sum<int>() with nan failed" );

   test_minmax<float>("float");
   test_minmax<double>("double");
   test_minmax<int>("int");
   test_minmax<short>("short");
   test_minmax<char>("char");

   test_align1();
   test_align2();
   test_avg();
   test_variance();
}

template<typename T> void test_minmax(const string& type)
{
   cout << "Testing min/max<"<<type<<">()" << endl;
   MArray<T,1> A(33);
   A = indexPosInt(A, 1);
   A(22) += T(100);
   T s = max(A);
   LTL_EXPECT_(A(22), s, "max<"<<type<<">() failed");
   A(23) = T(0);
   s = min(A);
   LTL_EXPECT_(A(23), s, "min<"<<type<<">() failed");
}

void test_align1(void)
{
   cout << "Testing alignment for float ... " << endl;
   MArray<float,1> A(100);
   A = indexPosInt(A, 1);

   const int align=4;
   for (int j=0; j<=align; ++j)
   {
      cout << "  testing alignment " << j << endl;
      float t = sum( A( Range(1+j,53+j) ) );
      float s=0;
      for (int i=1; i<=53; ++i)
         s += A(i+j);
      LTL_EXPECT_( s, t, "Alignment test failed for sum<float>");
   }
}

void test_align2(void)
{
   cout << "Testing alignment for double ... " << endl;
   MArray<double,1> A(100);
   A = indexPosInt(A, 1);

   const int align=2;
   for (int j=0; j<=align; ++j)
   {
      cout << "  testing alignment " << j << endl;
      double t = sum( A( Range(1+j,53+j) ) );
      double s=0;
      for (int i=1; i<=53; ++i)
         s += A(i+j);
      LTL_EXPECT_( s, t, "Alignment test failed for sum<double>");
   }
}

void test_avg(void)
{
   cout << "Testing averages ... " << endl;
   MArray<float,1> A(100);
   A = indexPosInt(A, 1);

   float a = average(A);
   LTL_EXPECT_( 50.5, a, "Vectorized average<float> test failed");
   a = average(A,1.0f);
   LTL_EXPECT_( 51, a, "Vectorized average_nan<float> test failed");

   MArray<double,1> D(100);
   D = indexPosInt(D, 1);
   double b = average(D);
   LTL_EXPECT_( 50.5, b, "Vectorized average<double> test failed");
   b = average(D,1.0);
   LTL_EXPECT_( 51, b, "Vectorized average_nan<double> test failed");
}

void test_variance(void)
{
   cout << "Testing variances ... " << endl;
   MArray<float,1> A(100);
   A = indexPosInt(A, 1);
   {
      float s=0, a=0;
      for (int i=1; i<=(int)A.nelements(); ++i)
         a += A(i);
      a /= A.nelements();
      for (int i=1; i<=(int)A.nelements(); ++i)
         s += pow2(A(i)-a);
      s /= (A.nelements()-1);
      cout << variance(A) << "," << s << endl;
      LTL_ASSERT_( fabs(variance(A)-s)<1e-4, "Vectorized variance<float> test failed");

      s=0, a=0;
      int n=0;
      for (int i=1; i<=(int)A.nelements(); ++i)
         if (A(i)!=1.0f)
         {
            a += A(i);
            ++n;
         }
      a /= n;
      n = 0;
      for (int i=1; i<=(int)A.nelements(); ++i)
         if (A(i)!=1.0f)
         {
            s += pow2(A(i)-a);
            ++n;
         }
      s /= (n-1);
      cout << variance(A,1.0f) << "," << s << endl;
      LTL_ASSERT_( fabs(variance(A,1.0f)-s)<1e-4, "Vectorized variance_nan<float> test failed");
   }
   {
      MArray<double,1> D(100);
      D = indexPosInt(D, 1);

      double s=0, a=0;
      for (int i=1; i<=(int)D.nelements(); ++i)
         a += D(i);
      a /= D.nelements();
      for (int i=1; i<=(int)D.nelements(); ++i)
         s += pow2(D(i)-a);
      s /= (D.nelements()-1);
      cout << variance(D) << "," << s << endl;
      LTL_ASSERT_( fabs(variance(D)-s)<1e-6, "Vectorized variance<double> test failed");

      s=0, a=0;
      int n=0;
      for (int i=1; i<=(int)D.nelements(); ++i)
         if (D(i)!=1.0f)
         {
            a += D(i);
            ++n;
         }
      a /= n;
      n=0;
      for (int i=1; i<=(int)D.nelements(); ++i)
         if (D(i)!=1.0f)
         {
            s += pow2(D(i)-a);
            ++n;
         }
      s /= (n-1);
      cout << variance(D,1.0) << "," << s << endl;
      LTL_ASSERT_( fabs(variance(D,1.0)-s)<1e-6, "Vectorized variance_nan<double> test failed");
   }
}

