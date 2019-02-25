/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testfitsio.cpp 363 2008-07-10 15:09:44Z drory $
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

//#define LTL_RANGE_CHECKING

#include <fitsio.h>
#include <statistics.h>
#include <marray_io.h>

#include <iostream>
#include <cstdio>

using namespace ltl;
using namespace util;

using std::cout;
using std::endl;

int main(int argc, char **argv);

void testextensionio( void )
{
   MArray<float,2> A(512,512), C;
   A = indexPosFlt(A,1) + 100.0*indexPosFlt(A,2);
   C = A+1.0f;
   
   FitsExtensionOut feo("testextensionio.fits");
   
   feo.addValueCard("TESTTEST", "This is a test");
   LTL_ASSERT_( feo.getString( "TESTTEST" ) == "This is a test",
                "FITS AddValueCard( string ) failed" );
   feo.addValueCard("TEST1234", 1.234);
   LTL_ASSERT_( feo.getFloat( "TEST1234" ) == 1.234,
                "FITS AddValueCard( float ) failed" ); 
   feo << emptyData; // primary with no data
   feo << A;         // 1st extension with data
   feo << emptyData; // 2nd extension with no data
   feo << C;  // 3rd extension with data

   FitsExtensionIn fei("testextensionio.fits");

   LTL_ASSERT_( fei.getString( "TESTTEST" ) == "This is a test",
                "FITS getString() failed within primary header" );
   LTL_ASSERT_( fei.getFloat( "TEST1234" ) == 1.234,
                "FITS getFloat() failed within primary header" );
   try
   {      
      while(true)
      {
         FitsIn fi(fei.getNextExtension());
         if( fei.getExtNo() % 2 )
         {         
            MArray<float,2> B;
            fi >> B;
            LTL_ASSERT_( allof(A == B),
                         "FITS extension data read or write failed");
            A += 1.0f;
         }
         LTL_ASSERT_( fi.getString( "TESTTEST" ) == "This is a test",
                      "FITS getString() failed within some extension header" );
         LTL_ASSERT_( fi.getFloat( "TEST1234" ) == 1.234,
                      "FITS getFloat() failed within some extension header" );
      }
   }
   catch(FitsException e)
   {
      if(e.what() != string("no further extensions")) throw;
      //else cerr << "ending with: " << e.what() << endl;
   }

   remove("testextensionio.fits");
}

int main(int argc, char **argv)
{
   cerr << "Testing MArray FITS extensions I/O ..." << endl;   
   const int offtsize = sizeof(off_t);
   if(sizeof(off_t)!=8){
     cerr << "Wrong offset type off_t: Must be 64 bit, but is "
	  << offtsize * 8 << " bit."<< endl;
     throw FitsException("");
   }
   try
   {
      testextensionio();
   }
   catch (exception &e){
      cerr << e.what() << endl;
      throw;
   }
}
