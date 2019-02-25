/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmarrayfvec.cpp 476 2010-11-12 06:00:58Z drory $
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

//#define LTL_TEMPLATE_LOOP_LIMIT 0
#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/fvector.h>
#include <ltl/statistics.h>
#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;


int main(int argc, char **argv)
{
   cerr << "Testing MArray<FVector<float,3>,N> ..." << endl;
   
   MArray<FVector<float,3>, 1> A(10);
   MArray<FVector<bool,3>, 1> B(10);
   FVector<float,3> v, w, zero;
   v = 1,2,3;
   w = 3,6,9;
   A = 2.0f*v + v;
   zero = 0;

   A = A;
   A = A+A;
   A = A+2*A;

   A = w;
   A = 2*A-2*w;
   for (int i=1; i<=A.length(1); ++i)
      LTL_ASSERT_( allof(A(i)==0.0f), "MArray<FVector<float,3>,1> basic expression/assignment failed." );
   for (int i=1; i<=A.length(1); ++i)
      LTL_ASSERT_( allof(A(i)==zero), "MArray<FVector<float,3>,1> basic expression/assignment failed." );
//   B = (A==0.0f);
//   B = (A==zero);
//   cout << B << endl;
//   LTL_ASSERT_( noneof(A), "MArray<FVector<float,3>,1> basic expression/assignment failed." );
//   LTL_ASSERT_( allof(B), "MArray<FVector<float,3>,1> basic expression/assignment failed." );
}

