/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testcomplex.cpp 515 2013-02-05 15:19:52Z cag $
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
#include <ltl/fvector.h>
#include <ltl/fmatrix.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>
#include <iomanip>

using namespace ltl;

using std::cout;
using std::endl;
using std::setprecision;

template<typename T>
void test_complex(void);
template<typename T>
void test_fv_complex(void);
template<typename T>
void test_fm_complex(void);

complex<float> user_func( complex<float> x )
{
   return x*x;
}
DECLARE_UNARY_FUNC( user_func, complex<float> )

#if defined(__GNUC__)
#  define THIS_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
   /* Test for GCC  3.6.x */
#  if (THIS_GCC_VERSION >= 40600 && THIS_GCC_VERSION < 40700)
#    define GCC_46_COMPLEX_DOUBLE_BUG
#  endif
#endif

int main(int argc, char **argv)
{
   cerr << "Testing MArray complex math ..." << endl;

    // this one test causes g++ to segfault on macosx intel with the apple-supplied g++=4.0 and 4.2
    // so we won't even try ...
#if defined(__APPLE__) && defined(__GNUC__) && !defined(__llvm__) && (__GNUC__==4 && __GNUC_MINOR__ < 3)
   cout << "Not testing test_complex<float> due to compiler bug." << endl;
#else
   test_complex<float>();
#endif
   test_complex<double>();

#ifdef GCC_46_COMPLEX_DOUBLE_BUG
   cout << "Not testing test_complex<double> due to gcc 4.6 compiler bug." << endl;
#endif

   cerr << "Testing FVector complex math ..." << endl;
   test_fv_complex<float>();
#ifndef GCC_46_COMPLEX_DOUBLE_BUG
   test_fv_complex<double>();
#endif

   cerr << "Testing FMatrix complex math ..." << endl;
   test_fm_complex<float>();
#ifndef GCC_46_COMPLEX_DOUBLE_BUG
   test_fm_complex<double>();
#endif

   cerr << "Testing user-suppplied function complex math ..." << endl;
   MArray<complex<float>,1> A(100), B(100);
   A = indexPosInt(A,1);
   B = user_func(A);
   LTL_ASSERT_( allof(B==A*A)," MArray user-supplied function on complex<T> failed" );
   return 0;
}


template<typename T>
void test_complex(void)
{
   MArray<complex<T>,1> A(10), B(10);
   A = complex<T>(1.0,2.0);
   complex<T> c(1.0,2.0);
   LTL_ASSERT_( allof(A==c)," MArray complex<T> assignment expression failed" );

   A = A + complex<T>(1.0);
   c = complex<T>(2.0,2.0);
   LTL_ASSERT_( allof(A==c)," MArray complex<T> simple add expression failed" );

   A = conj( A + complex<T>(1.0,-1.0) );
   c = complex<T>(3.0,-1.0);
   LTL_ASSERT_( allof(A==c)," MArray complex<T> conj() expression failed" );

   c = complex<T>(1.0,1.0);
   B = c;
   MArray<double,1> C;
   C = abs(B);
   LTL_ASSERT_( allof(C==abs(c))," MArray complex<T> abs() expression failed" );

   c = A(1);

   A = cos(  A + T(1.0) );
   A = exp(  A * A );
   A = log(  A + T(1.0) );
   A = sqrt( A + T(1.0) );
   A = tanh( A + T(1.0) );

   c = cos(  c + T(1.0) );
   c = exp(  c * c );
   c = log(  c + complex<T>(1.0) );
   c = sqrt( c + complex<T>(1.0) );
   c = tanh( c + complex<T>(1.0) );

   LTL_ASSERT_( allof(A==c)," MArray complex<T> transcendental expression failed" );
}

template<typename T>
void test_fv_complex(void)
{
   FVector<complex<T>,3> A, B;
   A = complex<T>(1.0,2.0);
   complex<T> c(1.0,2.0);
   LTL_ASSERT_( allof(A==c)," FVector complex<T> assignment expression failed" );

   A = A + complex<T>(1.0);
   c = c + complex<T>(1.0);
   LTL_ASSERT_( allof(A==c)," FVector complex<T> simple add expression failed" );

   A = conj( A + complex<T>(1.0,-1.0) );
   c = conj( c + complex<T>(1.0,-1.0) );
   LTL_ASSERT_( allof(A==c)," FVector complex<T> conj() expression failed" );

   c = complex<T>(1.0,1.0);
   B = c;
   FVector<T,3> C;
   C = abs(B);
   LTL_ASSERT_( allof(C==abs(c))," FVector complex<T> abs() expression failed" );

   c = A(1);

   A = cos(  A + T(1.0) );
   A = exp(  A * A );
   A = log(  A + T(1.0) );
   A = sqrt( A + T(1.0) );
   A = tanh( A + T(1.0) );

   c = cos(  c + T(1.0) );
   c = exp(  c * c );
   c = log(  c + T(1.0) );
   c = sqrt( c + T(1.0) );
   c = tanh( c + T(1.0) );

   LTL_ASSERT_( allof(A==c)," FVector complex<T> transcendental expression failed" );
}

template<typename T>
void test_fm_complex(void)
{
   FMatrix<complex<T>,3,4> A, B;
   A = complex<T>(1.0,2.0);
   complex<T> c(1.0,2.0);
   LTL_ASSERT_( allof(A==c)," FMatrix complex<T> assignment expression failed" );

   A = A + complex<T>(1.0);
   c = complex<T>(2.0,2.0);
   LTL_ASSERT_( allof(A==c)," FMatrix complex<T> simple add expression failed" );

   A = conj( A + complex<T>(1.0,-1.0) );
   c = complex<T>(3.0,-1.0);
   LTL_ASSERT_( allof(A==c)," FMatrix complex<T> conj() expression failed" );

   c = complex<T>(1.0,1.0);
   B = c;
   FMatrix<T,3,4> C;
   C = abs(B);
   LTL_ASSERT_( allof(C==abs(c))," FMatrix complex<T> abs() expression failed" );

   c = A(1,1);

   A = cos(  A + T(1.0) );
   A = exp(  A * A );
   A = log(  A + T(1.0) );
   A = sqrt( A + T(1.0) );
   A = tanh( A + T(1.0) );

   c = cos(  c + T(1.0) );
   c = exp(  c * c );
   c = log(  c + T(1.0) );
   c = sqrt( c + T(1.0) );
   c = tanh( c + T(1.0) );

   LTL_ASSERT_( allof(A==c)," FMatrix complex<T> transcendental expression failed" );
}
