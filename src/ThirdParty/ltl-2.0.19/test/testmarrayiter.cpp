/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmarrayiter.cpp 530 2014-02-07 14:50:28Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C) 2000-2002 Niv Drory <drory@usm.uni-muenchen.de>
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
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>
#include <algorithm>

using namespace ltl;

using std::cout;
using std::endl;

void test_stl_algo(void);
void test_ltl_iter(void);

int main(int argc, char **argv)
{
   cerr << "Testing stl <algorithm> with MArrays ..." << endl;
   test_stl_algo();
}

int increase (int i) {
  return ++i;
}

void test_stl_algo(void)
{
   MArray<int,1> A(10), B(10);
   A = 1,3,5,6,4,7,2,9,8,0;
   std::copy(A.begin(), A.end(), B.begin());
   LTL_ASSERT_( allof(A==B),
                "MArray stl::copy() failed" );

   B = indexPosInt(B,1)-1;
   std::sort(A.beginRA(), A.endRA());
   LTL_ASSERT_( allof(A==B),
                "MArray stl::sort() failed" );

   B = indexPosInt(B,1);
   std::transform(A.begin(), A.end(), A.begin(), increase);
   LTL_ASSERT_( allof(A==B),
                "MArray stl::transform() failed" );
}

void test_ltl_iter(void)
{
   
}

