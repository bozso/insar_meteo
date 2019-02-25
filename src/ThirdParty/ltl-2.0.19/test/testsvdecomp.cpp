/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testsvdecomp.cpp 363 2008-07-10 15:09:44Z drory $
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

#include <ltl/fmatrix/svdsolve.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
   cerr<<"Testing SV Decomposition ..."<<endl;
   FVector<double,4> X(2.), X0;
   X0 = X;
   FMatrix<double,4,4> A;
   A = 1., 2., 3., 2.,
      5., 11., 7., 13.,
      9., 7., 5., 1.,
      7., 13., 17., 11.;
   FVector<double,4> B(dot(A, X));

   X = 0.;
   X = SVDecomposition<double>::solve(A, B);
   
   cout << X << endl;
   LTL_ASSERT_( allof( (X - X0) < 1e-6 ) ,
                "SVDecomposition::solve() failed" );
   return 0;
}


/*
int main(int argc, char **argv)
{
   cerr<<"Testing SV Decomposition ..."<<endl;
   FVector<double,2> X(1.);
   FMatrix<double,4,2> A;
   A = 2,4,
       1,3,
       0,0,
       0,0;

   FVector<double,4> b(dot(A, X));   
   X = 0.;
   cout << A << endl;
   FVector<double,2> W;
   FMatrix<double,2,2> V;
   SVDecomposition<double>::svdcmp( A, W, V );
   X = SVDecomposition<double>::svbksb( A, W, V, b );

   cout << A << endl;
   cout << W << endl;
   cout << V << endl;
   //X = SVDecomposition<double>::solve(A, B);
   
   cout << X << endl;
   LTL_ASSERT_( allof( (X - 1.) < 1e-14 ) ,
                "SVDecomposition::solve() failed" );
   return 0;
}
*/
