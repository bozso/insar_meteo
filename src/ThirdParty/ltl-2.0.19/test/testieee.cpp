/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testieee.cpp 528 2013-12-09 17:30:54Z cag $
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

#include <ltl/marray.h>
#include <ltl/statistics.h>

#include <iostream>

using std::cerr;
using namespace ltl;

int main(int argc, char* argv[])
{

   cerr << "Testing presence of additional IEEE Math function(s) ...\n";
   
#ifdef HAVE_IEEE_MATH

   double s = 1.6;
   s = rint(s);

   MArray<double, 1> a(10);
   a = 1.6;
   a = rint(a);
   LTL_ASSERT_( allof(a == 2.0), "IEEE rint() failed" );

   a(5) = std::numeric_limits<double>::quiet_NaN();
   MArray<int, 1> i(10);
   i = 0;   
   i = isnan(a);
   LTL_ASSERT_( i(5), "IEEE isnan() failed" );

#else

   cerr << "Do not have IEEE Math functions.\n";

#endif

   return 0;
}
