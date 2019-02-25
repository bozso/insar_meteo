/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testlmfit.cpp 485 2011-06-15 20:03:15Z drory $
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

#include <ltl/nonlinlsqfit.h>
#include <ltl/marray_io.h>
#include <ltl/fvector.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::cerr;
using std::endl;

class function1
{
   public:
      typedef float value_type;
      float operator()( const float x, const FVector<double,3>& p, FVector<double,3>& df_dpi ) const
      {
         float value = p(1)*x*x + p(2)*x + p(3);
         df_dpi(1) = x*x;
         df_dpi(2) = x;
         df_dpi(3) = 1.0f;
         
         return value;
      }
};


class gauss1D
{
   public:
      typedef double value_type;
      double operator()( double x, const FVector<double,3>& P, FVector<double,3>& df_dpi ) const
      {
         double z = (x-P(2))/P(3);
         double e = exp( -0.5*z*z );
         double v = P(1)*e;
         df_dpi(1) = e;
         df_dpi(2) = P(1)/P(3)*e*z;
         df_dpi(3) = df_dpi(2)*z;
         return v;
      }
};

template<typename Solver>
void test1(void)
{
   function1 F;
   LMFit<function1, double, 3, Solver>M( F );

   MArray<float,1> X(10);
   X = 1.27355, -0.654883, 3.7178, 2.31818, 2.6652, -2.02182, 4.82368,
         -4.36208, 4.84084, 2.44391;

   MArray<float,1> Y(10);
   Y = 4.30655, 0.420333, 18.6992, 8.27967, 10.8136, 3.3598, 29.3496, 15.9234,
         29.4432, 9.6158;

   MArray<float,1> dY(10);
   dY = 1.0f;

   FVector<double,3> P0;
   P0 = 0.3, -1.0, 0.0;

   M.eval(X, Y, dY, -9999.0f, P0);

   FVector<double,3> P1;
   P1 = 1.01219, 0.985999, 0.996815;

   P0 = M.getResult();
   cout << "Expected : " << P1 << endl;
   cout << "Fitted   : " << P0 << endl;
   cout << "Errors   : " << M.getErrors() << endl;
   cout << "Chi2     : " << M.getChiSquare() << endl;
   cout << "Cov      : " << M.getCovarianceMatrix() << endl;

   P1 -= P0;
   LTL_ASSERT_( noneof( fabs(P1) > 1e-3 ), "LMFit failed!" );
}

template<typename Solver>
void test2(void)
{
   FVector<double,3> P, P1;
   FVector<double,3> dP;
   MArray<double,1> X(5), Y(5), dY(5);
   X = -2,-1,0,1,2;
   Y = exp( -0.5*X*X );
   dY = 0.1;
   P = 1.2, -0.1, 0.8;
   dP = 0.0;
   P1 = 1, 0, 1;
   cout << "Expected : " << P1  << endl;

   gauss1D F;
   LMFit<gauss1D, double, 3, Solver> M( F, 1000 );
   M.eval(X, Y, dY, -9999.0, P);
   P = M.getResult();
   
   cout << "Fitted   : " << P << endl;
   cout << "Errors   : " << M.getErrors() << endl;
   cout << "Chi2     : " << M.getChiSquare() << endl;
   cout << "Cov      : " << M.getCovarianceMatrix() << endl;
   P1 -= P;
   LTL_ASSERT_( noneof( fabs(P1) > 1e-3 ), "LMFit failed!" );
}

template<typename Solver>
void test3(void)
{
   function1 F;
   LMFit<function1, double, 3, Solver>M( F );

   MArray<float,1> X(10);
   X = 1.27355, -0.654883, 3.7178, 2.31818, 2.6652, -2.02182, 4.82368,
         -4.36208, 4.84084, 2.44391;

   for (int i=1; i<=X.length(1); ++i)
      X(i) += (-0.5+(rand()/(RAND_MAX + 1.0)))*0.1;

   MArray<float,1> Y(10);
   Y = 4.30655, 0.420333, 18.6992, 8.27967, 10.8136, 3.3598, 29.3496, 15.9234,
         29.4432, 9.6158;

   MArray<float,1> dY(10);
   dY = 1.0f;

   FVector<double,3> P0;
   P0 = 0.3, -1.0, 0.0;

   M.eval(X, Y, dY, -9999.0f, P0);

   FVector<double,3> P1;
   P1 = 1.01219, 0.985999, 0.996815;

   P0 = M.getResult();
   cout << "Expected : " << P1 << endl;
   cout << "Fitted   : " << P0 << endl;
   cout << "Errors   : " << M.getErrors() << endl;
   cout << "Chi2     : " << M.getChiSquare() << endl;
   cout << "Cov      : " << M.getCovarianceMatrix() << endl;
}

int main(int argc, char **argv)
{
   cerr << "Testing generic Lavenberg-Marquardt fit 1 using Gauss-Jordan ..." << endl;
   test1<GaussJ<double,3> >();
   cerr << "Testing generic Lavenberg-Marquardt fit 2 using Gauss-Jordan ..." << endl;
   test2<GaussJ<double,3> >();
   cerr << endl;
   cerr << "Testing generic Lavenberg-Marquardt fit 1 using SVD ..." << endl;
   test1<LUDecomposition<double,3> >();
   cerr << "Testing generic Lavenberg-Marquardt fit 2 using SVD ..." << endl;
   test2<LUDecomposition<double,3> >();
   cerr << endl;
   cerr << "Testing generic Lavenberg-Marquardt fit 3 using SVD ..." << endl;
   test3<LUDecomposition<double,3> >();
   cerr << "Testing generic Lavenberg-Marquardt fit 3 using using Gauss-Jordan ..." << endl;
   test3<GaussJ<double,3> >();
}
