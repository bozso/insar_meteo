/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: testmasimd2.cpp 476 2010-11-12 06:00:58Z drory $
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

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

#define N  47
#define TEST_RELATION( A, rel, B, C, relstr )  \
   C = (A rel B);                              \
   for( int i=1; i<=N; ++i )                   \
      LTL_ASSERT_( C(i), "Vectorizd relation "relstr" test failed" ); \
   cout << "Vectorized relation "relstr" test passed.\n";

#define TEST_LOGICAL( A, rel, B, C, relstr )  \
   C = (A rel B);                              \
   for( int i=1; i<=N; ++i )                   \
      LTL_ASSERT_( C(i)==(A(i) rel B(i)), "Vectorizd logical operation "relstr" test failed" ); \
   cout << "Vectorized logical operation "relstr" test passed.\n";

void test_double(void);
void test_float(void);
void test_int(void);
void test_short(void);
void test_char(void);

void test_bit_logical_char(void);
void test_bit_logical_short(void);
void test_bit_logical_int(void);

int main(int argc, char **argv)
{
   test_char();
   test_short();
   test_int();
   test_float();
   test_double();

   test_bit_logical_char();
   test_bit_logical_short();
   test_bit_logical_int();
}

void test_float(void)
{
   MArray<float,1> a(N), b(N);
   MArray<INT_TYPE(float),1> c(N);
   a = indexPosInt(a,1);
   b = indexPosInt(b,1)+1;
   TEST_RELATION( a, ==, a, c, "float ==");
   TEST_RELATION( a, !=, b, c, "float !=");
   TEST_RELATION( a, <, b, c, "float <");
   TEST_RELATION( b, >, a, c, "float >");
   for( int i=1; i<=N/2; ++i )
      b(i) -= 1;
   TEST_RELATION( a, <=, b, c, "float <=");
   TEST_RELATION( b, >=, a, c, "float >=");
}
void test_double(void)
{
   MArray<double,1> a(N), b(N);
   MArray<INT_TYPE(double),1> c(N);
   a = indexPosInt(a,1);
   b = indexPosInt(b,1)+1;
   TEST_RELATION( a, ==, a, c, "double ==");
   TEST_RELATION( a, !=, b, c, "double !=");
   TEST_RELATION( a, <, b, c, "double <");
   TEST_RELATION( b, >, a, c, "double >");
   for( int i=1; i<=N/2; ++i )
      b(i) -= 1;
   TEST_RELATION( a, <=, b, c, "double <=");
   TEST_RELATION( b, >=, a, c, "double >=");
}
void test_int(void)
{
   MArray<int,1> a(N), b(N), c(N);
   a = indexPosInt(a,1);
   b = indexPosInt(b,1)+1;
   TEST_RELATION( a, <, b, c, "int <");
   TEST_RELATION( b, >, a, c, "int >");
   TEST_RELATION( a, ==, a, c, "int ==");
   TEST_RELATION( a, !=, b, c, "int !=");
}
void test_short(void)
{
   MArray<short,1> a(N), b(N), c(N);
   a = indexPosInt(a,1);
   b = indexPosInt(b,1)+1;
   TEST_RELATION( a, <, b, c, "short <");
   TEST_RELATION( b, >, a, c, "short >");
   TEST_RELATION( a, ==, a, c, "short ==");
   TEST_RELATION( a, !=, b, c, "short !=");
}
void test_char(void)
{
   MArray<char,1> a(N), b(N), c(N);
   a = indexPosInt(a,1);
   b = indexPosInt(b,1)+1;
   TEST_RELATION( a, <, b, c, "char <");
   TEST_RELATION( b, >, a, c, "char >");
   TEST_RELATION( a, ==, a, c, "char ==");
   TEST_RELATION( a, !=, b, c, "char !=");
}

void test_bit_logical_char(void)
{
   MArray<char,1> A(N), B(N), C(N);
   A = indexPosInt(A,1);
   B = indexPosInt(B,1) + 3;
   TEST_LOGICAL( A, &, B, C, "char bitwise and" );
   TEST_LOGICAL( A, |, B, C, "char bitwise or" );
   TEST_LOGICAL( A, ^, B, C, "char bitwise xor" );
}
void test_bit_logical_short(void)
{
   MArray<short,1> A(N), B(N), C(N);
   A = indexPosInt(A,1);
   B = indexPosInt(B,1) + 3;
   TEST_LOGICAL( A, &, B, C, "short bitwise and" );
   TEST_LOGICAL( A, |, B, C, "short bitwise or" );
   TEST_LOGICAL( A, ^, B, C, "short bitwise xor" );
}
void test_bit_logical_int(void)
{
   MArray<int,1> A(N), B(N), C(N);
   A = indexPosInt(A,1);
   B = indexPosInt(B,1) + 3;
   TEST_LOGICAL( A, &, B, C, "int bitwise and" );
   TEST_LOGICAL( A, |, B, C, "int bitwise or" );
   TEST_LOGICAL( A, ^, B, C, "int bitwise xor" );
}
