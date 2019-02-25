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


#include <ltl/util/utdate.h>
#include <iostream>

using namespace std;
using namespace util;

int main()
{
   try
   {      
      cerr << "Testing util date methods:" << endl;

      cerr << "Now:" << endl;

      UTDate cnow;
      DCFDate dcfnow(cnow);
      FitsDate fitsnow(cnow);
      JulDate julnow(cnow);
   
      cerr << "ctime:    " << cnow.toString() << endl;
      cerr << "DCFDate:  " << dcfnow.toString() << endl;
      cerr << "FITSDate: " << fitsnow.toString() << endl;
      cerr << "JulDate:  " << julnow.toString() << endl;

      cerr << "Epoch:" << endl;

      UTDate cepoch( time_t(0) );
      DCFDate dcfepoch(cepoch);
      FitsDate fitsepoch(cepoch);
      JulDate julepoch(cepoch);
   
      cerr << "ctime:    " << cepoch.toString() << endl;
      cerr << "DCFDate:  " << dcfepoch.toString() << endl;
      cerr << "FITSDate: " << fitsepoch.toString() << endl;
      cerr << "JulDate:  " << julepoch.toString() << endl;

      cerr << "String Init for 4th March 1971, 18:10:00 UT" << endl;
   
      const string dcfbirth("D:04.03.71;T:6;U:19.10.00;    ");
      
      const string fitsbirthday("04/03/71");
      const string fitsbirth("1971-03-04T18:10:00");
      const double julbirth(2441015.25694444);
      cerr << "DCFDate:  " << UTDate(DCFDate(dcfbirth)).toString() 
           << " out of " << dcfbirth << endl;
      cerr << "FITSDate: " << UTDate(FitsDate(fitsbirth)).toString() 
           << " out of " << fitsbirth << endl;
      cerr << "day only: " << UTDate(FitsDate(fitsbirthday)).toString()
           << " out of " << fitsbirthday << endl;
      cerr << "JulDate:  " << UTDate(JulDate(julbirth)).toString()
           << " out of " << julbirth << endl;
   }
   catch(UTDateException ue)
   {
      cerr << ue.what() << endl;
      exit(1);
   }
   
   return 0;
}
