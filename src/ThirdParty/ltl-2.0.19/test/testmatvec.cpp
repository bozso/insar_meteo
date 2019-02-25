/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmatvec.cpp 363 2008-07-10 15:09:44Z drory $
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

//#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/fmatrix.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
   cerr<<"Testing Matrix-Vector products (template unrolled loops) ..."<<endl;
   FVector<float,3> A;
   FVector<float,2> C;
   C=0;
   FMatrix<float,2,3> M;
   A = 1.,2.,3.;
   M = 1,0,0,
       0,1,1;
   C = dot(M,A);

   LTL_ASSERT_( C(1)==1. && C(2) == 5., 
                "Matrix Vector product failed" );
   
   C = dot(M,2*A+A);
   LTL_ASSERT_( C(1)==3. && C(2) == 15., 
                "Matrix Vector-expression product failed" );

   C = dot((M+2*M)/3.,A);
   LTL_ASSERT_( C(1)==1. && C(2) == 5., 
                "Matrix-expression Vector product failed" );
   
   C = dot((M+2*M)/3.,A+A+A);
   LTL_ASSERT_( C(1)==3. && C(2) == 15., 
                "Matrix-expression Vector-expression product failed" );
   

   C = dot((M+2*M)/3.,A+A+A);
   LTL_ASSERT_( noneof( C - dot((M+2*M)/3.,A+A+A) ),
                "Matrix-expression Vector-expression product failed" );
   
   
   return 0;
}

