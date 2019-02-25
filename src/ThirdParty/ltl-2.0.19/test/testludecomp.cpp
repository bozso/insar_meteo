/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testludecomp.cpp 484 2011-06-10 16:36:52Z drory $
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

#include <ltl/fmatrix/lusolve.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
   cerr<<"Testing LU Decomposition ..."<<endl;
   FVector<double,4> X(1.);
   FMatrix<double,4,4> A;
   A = 1., 2., 3., 2.,
      5., 11., 7., 13.,
      9., 7., 5., 1.,
      7., 13., 17., 11.;
   FVector<double,4> B(dot(A, X));   
   X = 0.;
   X = LUDecomposition<double, 4>::solve(A, B);
   
   cout << X << endl;
   LTL_ASSERT_( allof( fabs(X - 1.) < 1e-14 ) ,
                "LUDecomposition::solve() failed" );

   FMatrix<double,4,4> Ainv, Id, C;
   Id = 0.0;
   Id.traceVector() = 1.0;

   Ainv = LUDecomposition<double, 4>::invert(A);
   C = dot(A,Ainv);
   cout << C << endl;
   LTL_ASSERT_( allof( fabs(C-Id) < 1e-14 ),
                "LUDecomposition::invert() failed" );

   return 0;
}
