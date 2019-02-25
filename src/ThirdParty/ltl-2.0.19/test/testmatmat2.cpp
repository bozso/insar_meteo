/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmatmat2.cpp 363 2008-07-10 15:09:44Z drory $
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

#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/fmatrix.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv)
{
   cerr<<"Testing Matrix-matrix products (for-loops) ..."<<endl;



   FMatrix<float,3,3> A;
   FMatrix<float,3,3> B;
   FMatrix<float,3,3> C;
   
   A = 0;
   B = 0;
   C = 1;
   
   A.traceVector() = 1;
   B.traceVector() = 1;

   C = dot(A,B);

   LTL_ASSERT_( allof(C == A),
               "Multiplication of Unit-Matrix failed" );
   
   FMatrix<float,2,3> AA;
   FMatrix<float,3,4> BB;
   FMatrix<float,2,4> CC;
   FMatrix<float,3,3> DD;
   
   AA = 1,2,3,
        4,5,6;
   BB = 1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12;
   
   CC = 38,44, 50, 56,
        83,98,113,128;
   
   LTL_ASSERT_( allof(CC == dot(AA,BB)),
               "Matrix multiplication failed" );
   
   LTL_ASSERT_( allof(CC+CC == dot(AA+AA,BB)),
               "Matrix expression multiplication failed" );
   
   LTL_ASSERT_( allof(CC*4 == dot(AA+AA,2*BB)),
               "Matrix expression multiplication failed" );
   
   DD = 17,22,27,
        22,29,36,
        27,36,45;

   LTL_ASSERT_( allof(DD == dot(transpose(AA),AA) ),
               "Matrix transpose multiplication failed" );
   
   

   return 0;
}

