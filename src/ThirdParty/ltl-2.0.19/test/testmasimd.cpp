/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: testmasimd.cpp 562 2015-04-30 16:01:16Z drory $
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

/*
 * g++ -O3 -msse3 -framework Accelerate -I. -I../ testmasimd.cpp
 */
#define LTL_RANGE_CHECKING
#define LTL_USE_SIMD
#define LTL_DEBUG_EXPRESSIONS
//#define LTL_UNROLL_EXPRESSIONS_SIMD

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

#define N  1024

#define TEST_ARITHMETIC_FLT( A, rel, B, C, relstr )                    \
   C = (A rel B);                                                      \
   for( int i=1; i<=(int)C.nelements(); ++i )                          \
      if (fabs(C(i)-(A(i) rel B(i)))>1e-6)                             \
      {                                                                \
         cout << i << " : " << C(i) << " != " << A(i) << relstr << B(i) << " = " << (A(i) rel B(i)) << endl; \
         cout << "Vectorizd float arithmetic operation " << relstr << " test failed";                              \
            abort();                                                   \
      }                                                                \
   cout << "Vectorized float arithmetic operation "relstr" test passed.\n";

#define TEST_ARITHMETIC_INT( A, rel, B, C, relstr )                    \
   C = (A rel B);                                                      \
   for( int i=1; i<=(int)C.nelements(); ++i )                          \
      if (abs(C(i)-(A(i) rel B(i)))!=0)                              \
      {                                                                \
         cout << i << " : " << C(i) << " != " << A(i) << relstr << B(i) << " = " << (A(i) rel B(i)) << endl; \
         cout << "Vectorizd integer arithmetic operation " << relstr << " test failed";                              \
            abort();                                                   \
      }                                                                \
   cout << "Vectorized integer arithmetic operation "relstr" test passed.\n";

void test_float(void);
void test_double(void);
void test_align1(void);
void test_align2(void);
void test_align3(void);

template <typename T>
void test_logic(void);

template <typename T>
void test_arith_int(void);

template <typename T>
void test_arith_float(void);

template <typename T>
void test_strange(void)
{
   MArray<T,1> B(4), C(5);
   C = 0;
   MArray<T,1> BB(4);
   BB = 65, 115, 165, 215;
   B = 65, 115, 165, 215;
   cout << B << endl;
   LTL_ASSERT( allof(B==BB), "Partial reduction sum failed");
}

int main(int argc, char **argv)
{
   test_arith_int<char>();
   test_arith_int<short>();
   test_arith_int<int>();
   test_arith_float<float>();
   test_arith_float<double>();

   test_float();
   test_double();
   test_align1();
   test_align2();
   test_align3();

   test_logic<char>();
   test_logic<short>();
   test_logic<int>();
   test_logic<float>();
   test_logic<double>();

   test_strange<float>();
   test_strange<double>();
   test_strange<int>();
   test_strange<short>();
   test_strange<char>();
}

void test_float(void)
{
   cout << "Testing float SIMD vectorized math ..." << endl;
   MArray<float,1> A(N), B(N), C(N);
   float *Ac = new float[N];
   float *Bc = new float[N];
   float *Cc = new float[N];

   cout << "---- B = 6.43243f;" << endl;
   B = 6.43243f;
   cout << "---- C = -1.43786f;" << endl;
   C = -1.43786f;
   for (int j=0; j<N; ++j)
   {
      Bc[j] = B(j+1);
      Cc[j] = C(j+1);
   }

   cout << "---- A = B + B/C" << endl;
   A = B + B/C;
   for (int j=0; j<N; ++j)
      Ac[j] = Bc[j] + Bc[j]/Cc[j];
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5f,
                  "Vector and scalar results differ in basic arithmetic");

   cout << "---- A = sqrt(-C) + sqrt(B);" << endl;
   A = sqrt(-C) + sqrt(B);
   for (int j=0; j<N; ++j)
      Ac[j] = sqrt(-Cc[j]) + sqrt(Bc[j]);
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5f,
                  "Vector and scalar results differ in mathlib arithmetic");

   cout << "---- A = pow2(-C) + pow3(B);" << endl;
   A = pow2(-C) + pow3(B);
   for (int j=0; j<N; ++j)
      Ac[j] = pow2(-Cc[j]) + pow3(Bc[j]);
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5f,
                  "Vector and scalar results differ in mathlib arithmetic");

   cout << "---- A = C - fabs(C);" << endl;
   A = C - fabs(C);
   for (int j=0; j<N; ++j)
      Ac[j] = Cc[j] - fabs(Cc[j]);
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5f,
                  "Vector and scalar results differ in ltl-defined arithmetic");

#ifdef HAVE_APPLE_VECLIB
   cout << "Testing Apple vecLib" << endl;
   cout << "---- A = log1p(B);" << endl;
   A = log1p(B);
   for( int j=0; j<N; ++j )
      Ac[j] = log1p(Bc[j]);
   for( int j=0; j<N; ++j )
      LTL_ASSERT_( fabs(Ac[j] - A(j+1)) < 1e-5f, "Vector and scalar results differ in vecLib arithmetic" );
#endif

   delete[] Ac;
   delete[] Bc;
   delete[] Cc;
}

void test_double(void)
{
   cout << "Testing double SIMD vectorized math ..." << endl;
   MArray<double,1> A(N), B(N), C(N);
   double *Ac = new double[N];
   double *Bc = new double[N];
   double *Cc = new double[N];

   cout << "---- B = 6.43243" << endl;
   B = 6.43243;
   cout << "---- C = -1.43786" << endl;
   C = -1.43786;
   for (int j=0; j<N; ++j)
   {
      Bc[j] = B(j+1);
      Cc[j] = C(j+1);
   }

   cout << "---- A = B + B/C" << endl;
   A = B + B/C;
   for (int j=0; j<N; ++j)
      Ac[j] = Bc[j] + Bc[j]/Cc[j];
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5,
                  "Vector and scalar differ in basic double arithmetic");

   cout << "---- A = sqrt(-C) + sqrt(B);" << endl;
   A = sqrt(-C) + sqrt(B);
   for (int j=0; j<N; ++j)
      Ac[j] = sqrt(-Cc[j]) + sqrt(Bc[j]);
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5,
                  "Vector and scalar differ in mathlib double arithmetic");

   cout << "---- A = C - fabs(C);" << endl;
   A = C - fabs(C);
   for (int j=0; j<N; ++j)
      Ac[j] = Cc[j] - fabs(Cc[j]);
   for (int j=0; j<N; ++j)
      LTL_ASSERT_(fabs(Ac[j] - A(j+1)) < 1e-5,
                  "Vector and scalar differ in ltl-defined double arithmetic");

   delete[] Ac;
   delete[] Bc;
   delete[] Cc;
}

void test_align1(void)
{
   cout << "Testing alignment for float ... " << endl;
   MArray<float,1> A(100);
   A = indexPosInt(A, 1);

   MArray<float,1> B(100);
   B = 0;
   const int align=4;
   for (int j=0; j<=align; ++j)
   {
      cout << "  testing alignment " << j << endl;
      B(Range(1+align+j,53+align+j)) = A( Range(1+j,53+j) );
      for (int i=1; i<=53; ++i)
         LTL_ASSERT_(B(i+align+j) == A(i+j), "Alignment test failed for float");
   }
}

void test_align2(void)
{
   cout << "Testing alignment for double ... " << endl;
   MArray<double,1> A(100);
   A = indexPosInt(A, 1);

   MArray<double,1> B(100);
   B = 0;

   const int align=2;
   for (int j=0; j<=align; ++j)
   {
      cout << "  testing alignment " << j << endl;
      B(Range(1+align+j,53+align+j)) = A( Range(1+j,53+j) );
      for (int i=1; i<=53; ++i)
         LTL_ASSERT_(B(i+align+j) == A(i+j), "Alignment test failed double");
   }
}

void test_align3(void)
{
   cout << "Testing alignment for short ... " << endl;
   MArray<short,1> A(100);
   A = indexPosInt(A, 1);

   MArray<short,1> B(100);
   B = 0;

   const int align=8;
   for (int j=0; j<=align; ++j)
   {
      cout << "  testing alignment " << j << endl;
      B(Range(1+align+j,53+align+j)) = A( Range(1+j,53+j) );
      for (int i=1; i<=53; ++i)
         LTL_ASSERT_(B(i+align+j) == A(i+j), "Alignment test failed short");
   }
}

template <typename T>
void test_arith_int(void)
{
   const int K=47;
   MArray<T,1> A(K), B(K), C(K);

   A=T(2); B=T(3);

   TEST_ARITHMETIC_INT( A, +, B, C, "+");
   TEST_ARITHMETIC_INT( A, -, B, C, "-");
   TEST_ARITHMETIC_INT( A, *, B, C, "*");
   TEST_ARITHMETIC_INT( A, /, B, C, "/");
}


template <typename T>
void test_arith_float(void)
{
   const int K=47;
   MArray<T,1> A(K), B(K), C(K);

   A=T(2); B=T(3);

   TEST_ARITHMETIC_FLT( A, +, B, C, "+");
   TEST_ARITHMETIC_FLT( A, -, B, C, "-");
   TEST_ARITHMETIC_FLT( A, *, B, C, "*");
   TEST_ARITHMETIC_FLT( A, /, B, C, "/");
}


template<typename T>
void test_logic(void)
{
   cout << "Testing logical reductions ..." << endl;
   MArray<T,1>A(10);
   A = 0;

   LTL_ASSERT_( allof(A==A),
                "MArray allof() expression failed" );
   LTL_ASSERT_( noneof(A!=A),
                "MArray allof() expression failed" );
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

