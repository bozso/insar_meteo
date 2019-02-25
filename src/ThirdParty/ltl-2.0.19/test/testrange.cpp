/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id$
* ---------------------------------------------------------------------
*
* Copyright (C)  Jan Snigula <snigula@usm.uni-muenchen.de>
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
#include <ltl/marray_io.h>
#include <ltl/statistics.h>
#include <iostream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv);

void testrange( void )
{
   MArray<int,1> data(9);
   data = 1, 2, 3, 4, 5, 6, 7, 8, 9;

   MArray<int,1> res(3);
   res = 3, 4, 5;

   MArray<int,1> resp(3);
   resp = 5, 6, 7;

   MArray<int,1> resm(3);
   resm = 2, 3, 4;

   MArray<int,1> respe(3);
   respe = 6, 7, 8;

   MArray<int,1> resme(3);
   resme = 1, 2, 3;

   MArray<int,1> rese(3);
   rese = 7, 8, 9;

   Range R1(3,5);

   cerr << "Testing operator=...";
   Range RE1(7,9);
   Range RE2 = RE1;

   if( anyof(data(RE2) != rese) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else
      cerr << "passed" << endl;
   
   cerr << "Testing operator+...";
   Range RP = R1 + 2;

   if( anyof(data(RP) != resp) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else
      cerr << "passed" << endl;
   
   cerr << "Testing operator-...";
   Range RM = R1 - 1;

   if( anyof(data(RM) != resm) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else
      cerr << "passed" << endl;

   cerr << "Testing operator+=...";
   Range RPE(1,3);
   RPE += 5;

   if( anyof(data(RPE) != respe) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else
      cerr << "passed" << endl;

   cerr << "Testing operator-=...";
   Range RME(5,7);
   RME -= 4;

   if( anyof(data(RME) != resme) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else
      cerr << "passed" << endl;
}

int main(int argc, char **argv)
{
   cerr << "Testing dynamic Ranges ..." << endl;
   testrange();
}
