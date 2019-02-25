/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: teststatistics.cpp 502 2012-11-30 18:58:32Z cag $
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

#include <cfloat>

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>


using namespace ltl;

using std::cout;
using std::endl;

void test_sum(void);
void test_kappa(void);
void test_median(void);
void test_median_estimate(void);
void test_histogram(void);
void test_kappa_sigma(void);
void test_kappa_sigma_median(void);
void test_kappa_median_average(void);
void test_biweight(void);

int main(void)
{
   cerr << "Testing MArray statistics  ..." << endl;
   test_sum();
   test_kappa();
   test_median();
   test_median_estimate();
   test_histogram();
   test_kappa_sigma();
   test_kappa_sigma_median();
   test_kappa_median_average();
   test_biweight();
}

void test_biweight(void)
{
   MArray<float,2> A(2,4);
   A = 1., 2., 3., 4., 3., 2., 1., 100.;
   float m, s;
   s = robust_sigma (A);
   cout << "  robust_sigma () gave " << s << ", 1.26776 expected." << endl;
   LTL_ASSERT_ (fabs(s-1.26776)<1e-5, "robust_sigma () failed!")

   m = biweight_mean (A, &s);
   cout << "  biweight_mean () gave mean " << m << ", 2.27886 expected." << endl;
   cout << "  biweight_mean () gave sigma " << s << ", 1.23697 expected." << endl;
   LTL_ASSERT_ (fabs(m-2.27886)<1e-5, "biweight_mean () failed!")
   LTL_ASSERT_ (fabs(s-1.23697)<1e-5, "biweight_mean () failed!")
}

void test_kappa_sigma(void)
{
   double m=0.0, s=0.0;
   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   kappa_sigma_average( A, 3.0, &m, &s );
   kappa_sigma_average( A+1, 3.0, &m, &s );
   kappa_sigma_average( A, 3.0, 30.0, &m, &s );
   kappa_sigma_average( A+1, 3.0, 30.0, &m, &s );
}

void test_median_estimate(void)
{
   cerr << "Testing median estimates ... " << endl;

   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   double step;
   double m = median_estimate(A,30,1.0f,100.0f, &step);
   LTL_ASSERT_( m - 50.0f <= step, "median_estimate on MArray failed");
   LTL_ASSERT_( 50.0f - m <= step, "median_estimate on MArray failed");
   m = median_estimate(A+1.0f,30,2.0f,101.0f, &step);
   LTL_ASSERT_( m - 51.0f <= step, "median_estimate on Expression failed");
   LTL_ASSERT_( 51.0f - m <= step, "median_estimate on Expression failed");
   m = median_estimate(A,30,1.0f,100.0f, 0.0f, &step);
   LTL_ASSERT_( m - 50.0f <= step, "median_estimate NaN on MArray failed");
   LTL_ASSERT_( 50.0f - m <= step, "median_estimate NaN on MArray failed");
   m = median_estimate(A+1.0f,30,2.0f,101.0f, 1.0f, &step);
   LTL_ASSERT_( m - 51.0f <= step, "median_estimate NaN on Expression failed");
   LTL_ASSERT_( 51.0f - m <= step, "median_estimate NaN on Expression failed");
}

void test_histogram(void)
{
   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   MArray<float,1> H(10);
   H = 10;
   H(1) = 9;
   LTL_ASSERT_( allof(histogram( A, 10, 0.0f, 100.0f)==H), "histogram on MArray failed" );
   H(1) = 8;
   LTL_ASSERT_( allof(histogram( A+1, 10, 0.0f, 100.0f)==H), "histogram on Expression failed" );
   H(1) = 8;
   LTL_ASSERT_( allof(histogram( A, 10, 0.0f, 100.0f, 1.0f)==H), "histogram NaN on MArray failed" );
   H(1) = 7;
   LTL_ASSERT_( allof(histogram( A+1, 10, 0.0f, 100.0f, 2.0f)==H), "histogram NaN on Expression failed" );
}

void test_median(void)
{
   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   LTL_EXPECT_( 50.5, median_exact(A), "median_exact on MArray failed");
   LTL_EXPECT_( 51.5, median_exact(A+1), "median_exact on Expression failed");

   A(3) = -1.0f;
   A(57) = -1.0f;
   LTL_EXPECT_(50.5, median_exact( A, -1.0f), "median_exact on MArray with nan failed");
   LTL_EXPECT_(51.5, median_exact( A+1, 0.0f), "median_exact on Expression with nan failed");
}

void test_sum(void)
{
   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   int c = count(A-1);
   LTL_EXPECT_( 99, c, "count() failed" );
   c = count(A*A-1);
   LTL_EXPECT_( 99, c, "count() failed" );

   MArray<float,2> B(10,10);
   B = indexPosFlt(B,1) + indexPosFlt(B,2)*10;

   MArray<int,1> I(100);
   I = indexPosInt(I,1);

   float s;
   s = sum(A);
   LTL_EXPECT_( 100*101/2, s, "sum() failed" );

   s = sum(A*A+1);
   float s2 = 0;
   for(int i=1; i<=(int)A.nelements(); ++i)
      s2 += A(i)*A(i)+1;

   LTL_EXPECT_( s2, s, "sum() failed" );

   s = sum( B(Range::all(), 5) );
   LTL_EXPECT_( 555, s,
                "sum() with slice failed" );

   s = sum( B(5,Range::all()) );
   LTL_EXPECT_( 600, s,
                "sum() with slice and stride failed" );

   A(55) += 100;
   s = max(A);
   LTL_EXPECT_( A(55), s,
                "max<float>() failed" );
   A(55) = 0.0;
   s = min(A);
   LTL_EXPECT_( A(55), s,
                "min<float>() failed" );

   A(55) = 0.0;
   s = min(A, 0.0f);
   LTL_EXPECT_( A(1), s,
                "min<float>(nan) failed" );

   I(55) += 100;
   int i = max(I);
   LTL_EXPECT_( I(55), i,
                "max<int>() failed" );
   I(55) = 0;
   i = min(I);
   LTL_EXPECT_( I(55), i,
                "min<int>() failed" );
   }

void test_kappa(void)
{
   cerr << "Testing kapp-sigma averages ... " << endl;

   MArray<float,2> A(10, 10);
   A = indexPosFlt(A, 1) + 10 * (indexPosFlt(A, 2) - 1);

   MArray<float,2> B(A.shape());
   B = A;
   B(Range(5,6), Range(5,6)) = 0;

   LTL_ASSERT_( DBL_EPSILON >= fabs( (average(A) - 50.5) / 50.5 ),
   "average without nan failed");

   LTL_ASSERT_( DBL_EPSILON >= fabs( (average(B, 0.0f) - 50.5) / 50.5 ),
   "average with nan present failed");

   LTL_ASSERT_( DBL_EPSILON >= fabs( (average(B, -1.0f) - 48.48) / 48.48 ),
   "average with nan but not present failed");

   for (int k = 4; k > 1; --k)
   {
      double mean, sigma;
      double kappa = double(k);
      LTL_ASSERT_( 100 == kappa_sigma_average(A, kappa, &mean, &sigma),
                  "kappa sigma without nan did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma without nan did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - 29.011491975882) / 29.011491975882),
      "kappa sigma without nan did not yield correct sigma");
      LTL_ASSERT_( 96 == kappa_sigma_average(B, kappa, 0.0f, &mean, &sigma),
                  "kappa sigma with nan present did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma with nan present did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - 29.598008467854) / 29.598008467854),
      "kappa sigma with nan present did not yield correct sigma");
      LTL_ASSERT_( 100 == kappa_sigma_average(A, kappa, -1.0f, &mean, &sigma),
                  "kappa sigma with nan but not present did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma with nan but not present did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - 29.011491975882) / 29.011491975882),
      "kappa sigma with nan but not present did not yield correct sigma");
   }

   {
      double mean, sigma;
      LTL_ASSERT_( 2 == kappa_sigma_average(A, 1.0, &mean, &sigma),
                  "kappa sigma without nan did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma without nan did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - M_SQRT1_2) / M_SQRT1_2),
      "kappa sigma without nan did not yield correct sigma");
      LTL_ASSERT_( 2 == kappa_sigma_average(B, 1.0, 0.0f, &mean, &sigma),
                  "kappa sigma with nan present did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma with nan present did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - M_SQRT1_2) / M_SQRT1_2),
      "kappa sigma with nan present did not yield correct sigma");
      LTL_ASSERT_( 2 == kappa_sigma_average(A, 1.0, -1.0f, &mean, &sigma),
                  "kappa sigma with nan but not present did not yield correct remaining elements");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (mean - 50.5) / 50.5),
      "kappa sigma with nan but not present did not yield correct mean");
      LTL_ASSERT_( FLT_EPSILON >= fabs( (sigma - M_SQRT1_2) / M_SQRT1_2),
      "kappa sigma with nan but not present did not yield correct sigma");
   }
}

void test_kappa_sigma_median(void)
{
   cerr << "Testing kappa-sigma-median ..." << endl;

   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   double median=0.0, sigma=0.0;
   int n = kappa_sigma_median(A,2.8,&median,&sigma);
   LTL_EXPECT_( 100, n, "kappa_sigma_median on MArray failed");
   LTL_EXPECT_( 50.5, median, "kappa_sigma_median on MArray failed");

   n = kappa_sigma_median(A+1,2.8,&median,&sigma);
   LTL_EXPECT_( 100, n, "kappa_sigma_median on Expr failed");
   LTL_EXPECT_( 51.5, median, "kappa_sigma_median on Expr failed");

   n = kappa_sigma_median(A,2.8,50.0,&median,&sigma);
   LTL_EXPECT_( 99, n, "kappa_sigma_median_nan on MArray failed");
   LTL_EXPECT_( 51.0, median, "kappa_sigma_median_nan on MArray failed");

   n = kappa_sigma_median(A+1,2.8,40.0,&median,&sigma);
   LTL_EXPECT_( 99, n, "kappa_sigma_median_nan on Expr failed");
   LTL_EXPECT_( 52.0, median, "kappa_sigma_median_nan on Expr failed");
}

void test_kappa_median_average(void)
{
   cerr << "Testing kappa-median-average ..." << endl;

   MArray<float,1> A(100);
   A = indexPosFlt(A,1);

   double mean=0.0, sigma=0.0;
   int n = kappa_median_average(A,2.8,&mean,&sigma);
   LTL_EXPECT_( 100, n, "kappa_median_average on MArray failed");
   LTL_EXPECT_( 50.5, mean, "kappa_median_average on MArray failed");

   n = kappa_median_average(A+1,2.8,&mean,&sigma);
   LTL_EXPECT_( 100, n, "kappa_median_average on Expr failed");
   LTL_EXPECT_( 51.5, mean, "kappa_median_average on Expr failed");

   n = kappa_median_average(A,2.8,40.0,&mean,&sigma);
   LTL_EXPECT_( 99, n, "kappa_median_average on MArray failed");
   //LTL_EXPECT_( 50.6061, mean, "kappa_median_average on MArray failed");

   n = kappa_median_average(A+1,2.8,40.0,&mean,&sigma);
   LTL_EXPECT_( 99, n, "kappa_median_average on Expr failed");
   //LTL_EXPECT_( 52.5, mean, "kappa_median_average on Expr failed");
}
