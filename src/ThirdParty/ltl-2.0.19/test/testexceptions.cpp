/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testexceptions.cpp 363 2008-07-10 15:09:44Z drory $
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

#include <ltl/misc/exceptions.h>
#include <ltl/util/u_exception.h>

#include <iostream>
#include <exception>
#include <string>

using namespace std;

using namespace ltl;
using namespace util;

void testing(const string method)
{
   cerr << "testing " << method << " ... ";
   cerr.flush();
}
void show_ok(const bool is_ok)
{
   if(!is_ok)
      cerr << "not ";
   cerr << "ok" << endl;
}


int main(int argc, char **argv)
{
   bool ok = false;
   testing("std::exception");
   try
   {
      throw std::exception();
   }
   catch(std::exception& e)
   {
      ok = true;
   }
   catch(...)
   {
      cerr << "something else and " << endl;
   }   
   show_ok(ok);

   ok = false;
   testing("ltl::FitsException");
   try
   {
      throw FitsException("fits exception");
   }
   catch(FitsException fe)
   {
      ok = true;
   }
   show_ok(ok);

   ok = false;
   testing("util::UException");
   try
   {
      throw UTDateException("util exception");
   }
   catch(UTDateException ue)
   {
      ok = true;
   }
   show_ok(ok);

   ok = false;
   testing("util::UTDateException");
   try
   {
      throw UTDateException("utdate exception");
   }
   catch(UTDateException utde)
   {
      ok = true;
   }
   show_ok(ok);

   ok = false;
   testing("util::StringException");
   try
   {
      throw UTDateException("utdate exception");
   }
   catch(UTDateException utde)
   {
      ok = true;
   }
   show_ok(ok);

   return 0;
}
