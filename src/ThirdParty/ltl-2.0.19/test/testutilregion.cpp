/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testutildate.cpp 363 2008-07-10 15:09:44Z drory $
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


#include <ltl/util/region.h>
#include <iostream>

using namespace std;
using namespace util;

int main()
{
   try
   {      
      cerr << "Testing util region methods..." << endl;
      string teststring = "[1:10,21:40]";
      Region testregion(teststring, 2);
      string outstring = testregion.toString();
      if( testregion.toString() != teststring )
         throw UException( teststring + string(" does not match ") + outstring);
   }
   catch(UException ue)
   {
      cerr << ue.what() << endl;
      exit(1);
   }
   
   return 0;
}
