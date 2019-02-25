/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testgaussj2.cpp 484 2011-06-10 16:36:52Z drory $
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

#ifdef LTL_TEMPLATE_LOOP_LIMIT
#undef LTL_TEMPLATE_LOOP_LIMIT
#endif
#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/fmatrix/gaussj.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
   cerr<<"Testing Gauss-Jordan elimination (for loops) ..."<<endl;
   FVector<double,4> X(1.);
   FMatrix<double,4,4> A;
   A = 1., 2., 3., 2.,
      5., 11., 7., 13.,
      9., 7., 5., 1.,
      7., 13., 17., 11.;
   FVector<double,4> B(dot(A, X));

   FMatrix<double,4,4> AA(A);
   
   try
   {
         X = 0.;
         X = GaussJ<double, 4>::solve(A, B);
         GaussJ<double, 4>::eval(A, B);
   }
   catch(LinearAlgebraException e)
   {
      cout << e.what() << endl;
   }
   
   LTL_ASSERT_( allof( (X - 1.) < 1e-14 ) ,
                "GaussJ.solve() failed" );

   LTL_ASSERT_( allof( (B - 1.) < 1e-14 ) ,
                "solution of GaussJ.eval() failed" );

   FMatrix<double, 4, 4> E(dot(A, AA));
   E.traceVector() -= 1.;
   
   LTL_ASSERT_( allof( E < 1e-14 ) ,
                "matrix inversion of GaussJ.eval() failed" );
   
   FMatrix<double,4,4> Ainv, Id, C;
   Id = 0.0;
   Id.traceVector() = 1.0;

   Ainv = GaussJ<double, 4>::invert(AA);
   C = dot(AA,Ainv);
   LTL_ASSERT_( allof( fabs(C-Id) < 1e-13 ),
                "GaussJ::invert() failed" );
   return 0;
}
