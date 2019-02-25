/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testfitsio.cpp 545 2014-08-14 08:41:06Z drory $
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
void testfloat( void );
void testdouble( void );
void testbyte( void );
void testshort( void );
void testint( void );
void testcopy( void );
void testreadline( void );
void testwriteregion( void );
void testinmemoryread ( void );
void testinmemorywrite ( void );

void testfloat( void )
{
   cerr << "  BITPIX -32 ..." << endl;   

   MArray<float,2> A(512,512);

   A = indexPosFlt(A,1) + 100*indexPosFlt(A,2);
   
   FitsOut fo("test-32.fits");
   
   fo.addValueCard("TESTTEST", "This is a test");
   LTL_ASSERT_( fo.getString( "TESTTEST" ) == "This is a test",
                "FITS AddValueCard( string ) failed" );
   fo.addValueCard("TEST1234", 1.234);
   LTL_ASSERT_( fo.getFloat( "TEST1234" ) == 1.234,
                "FITS AddValueCard( float ) failed" ); 
   fo << A;

   MArray<float,2> B;
   FitsIn fi("test-32.fits");
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS -32 read or write failed");

   LTL_ASSERT_( fi.getString( "TESTTEST" ) == "This is a test",
                "FITS getString() failed" );
   LTL_ASSERT_( fi.getFloat( "TEST1234" ) == 1.234,
                "FITS getFloat() failed" );
}

void testdouble( void )
{
   cerr << "  BITPIX -64 ..." << endl;   
   MArray<double,2> A(512,512);

   A = indexPosDbl(A,1) + 25*indexPosDbl(A,2);
   FitsOut fo("test-64.fits");
   
   fo << A;

   MArray<double,2> B;
   FitsIn fi("test-64.fits");
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS -64 read or write failed");

   fi.freeData();
   remove("test-64.fits");
}

void testint( void )
{
   cerr << "  BITPIX 32 ..." << endl;   
   MArray<int,2> A(512,512);

   A = indexPosInt(A,1) + 25*indexPosInt(A,2);
   FitsOut fo("test32.fits");
   
   fo << A;

   MArray<int,2> B;
   FitsIn fi("test32.fits");
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS 32 read or write failed");

   fi.freeData();
   remove("test32.fits");
}

void testshort( void )
{
   cerr << "  BITPIX 16 ..." << endl;   
   MArray<short,2> A(512,512);

   A = indexPosInt(A,1) + 25*indexPosInt(A,2);
   FitsOut fo("test16.fits");
   
   fo << A;

   MArray<short,2> B;
   FitsIn fi("test16.fits");
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS 16 read or write failed");

   fi.freeData();
   remove("test16.fits");
}

void testbyte( void )
{
   cerr << "  BITPIX 8 ..." << endl;   
   MArray<char,2> A(16,16);

   A = indexPosInt(A,1) + 25*indexPosInt(A,2);
   FitsOut fo("test8.fits");
   
   fo << A;

   MArray<char,2> B;
   FitsIn fi("test8.fits");
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS 8 read or write failed");

   fi.freeData();
   remove("test8.fits");
}

void testcopy( void )
{
   cerr << "  BITPIX -32 copy ..." << endl;   

   FitsIn fi1("test-32.fits");
   FitsOut fo("testc.fits", fi1);
   fi1 >> fo;

   MArray<float,2> A, B;
   fi1 >> A;
   FitsIn fi3("testc.fits");
   fi3 >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS -32 copy failed");

   LTL_ASSERT_( fi3.getString( "TESTTEST" ) == "This is a test",
                "FITS getString() failed" );
   LTL_ASSERT_( fi3.getFloat( "TEST1234" ) == 1.234,
                "FITS getFloat() failed" ); 

   fi3.freeData();
   remove("testc.fits");
}

void testreadline( void )
{
   cerr << "  Reading file per line ..." << endl;
   
   MArray<float, 2> A;
   MArray<float, 1> B(512);

   FitsIn fi("test-32.fits");
   fi >> A;

   for(size_t i = 1; i <= size_t(fi.getNaxis(2)); ++i)
   {
      fi >> B;
      LTL_ASSERT_( allof(A(Range::all(), i) == B),
                   "FITS reading per line failed");
   }

   fi.freeData();
   remove("test-32.fits");
}

void testwriteregion( void )
{
   cerr << "  Writing region of file ..." << endl;   

   MArray<float,2> A(500,500);

   A = indexPosInt(A,1) + 100*indexPosInt(A,2);
   
   FitsOut fo("test-reg.fits");
   Region region(2);
   region.setRange(1, 1, 1000);
   region.setRange(2, 1, 1000);
   // creating file
   fo.setGeometry(-32, region);

   region.setRange(1, 251, 750);
   region.setRange(2, 251, 750);
   fo.setRegion(region);
   fo << A;

   MArray<float,2> B;
   FitsIn fi("test-reg.fits", region);
   fi >> B;

   LTL_ASSERT_( allof(A == B),
                "FITS file region read or write failed (written values)");
   B.free();

   region.setRange(1, 751, 1000);
   region.setRange(2, 1, 1000);
   fi.setRegion(region);
   fi >> B;

   LTL_ASSERT_( allof(B == 0.0f),
                "FITS file region read or write failed (truncate with zeroes)");   

   fi.freeData();
   remove("test-reg.fits");
}

void testwriteconstarray( void )
{
   MArray<float,2> A(500,500);
   A = indexPosInt(A,1) + 100*indexPosInt(A,2);
   const MArray<float,2>B(A);
   FitsOut fo("testconst.fits");
   fo << B;
   remove("testconst.fits");
}

void testextensionio( void )
{
   FitsExtensionOut feo("testextensionio.fits");
   feo << emptyData;
   remove("testextensionio.fits");
}

void testinmemoryread( void )
{
   cerr << "  Testing in-memory read ..." << endl;

   MArray<float,2> A(2448,2050);

   A = indexPosInt(A,1) + 25*indexPosInt(A,2);
   FitsOut fo("test_inmem_float.fits");

   fo << A;

   int fd = open("test_inmem_float.fits", O_RDONLY);
   if (fd < 0)
   {
	  cerr << "cannot open test_inmem_float.fits" << endl;
      throw FitsException("");
   }

   struct stat fdstat;
   fstat(fd, &fdstat);

   float*	buf = new float[fdstat.st_size];
   size_t	bytes = read(fd, buf, fdstat.st_size);
   if ((off_t)bytes != fdstat.st_size)
   {
	  cerr << "cannot read test_inmem_float.fits" << endl;
      throw FitsException("");
   }
   close(fd);

   ltl::FitsIn     fitsIn((unsigned char*)buf, bytes, true);
   ltl::MArray<float,2>	imageArray;

   fitsIn >> imageArray;

   LTL_ASSERT_( fitsIn.getNaxis() == 2, "FITS in-memory Naxis wrong" );
   LTL_ASSERT_( fitsIn.getBitpix() == -32, "FITS in-memory bitpix wrong" );
   LTL_ASSERT_( fitsIn.getNaxis(1) == 2448, "FITS in-memory Naxis1 wrong" );
   LTL_ASSERT_( fitsIn.getNaxis(2) == 2050, "FITS in-memory Naxis1 wrong" );

   LTL_ASSERT_( allof(A == imageArray), "FITS in-memory data mismatch");

   delete [] buf;
   remove("test_inmem_float.fits");
}

void testinmemorywrite( void )
{
   cerr << "  Testing in-memory write ..." << endl;
   MArray<float,2> A(2448,2050);

   A = indexPosInt(A,1) + 25*indexPosInt(A,2);

   unsigned char*	buf = NULL;
   size_t			bufSize = 0;
   FitsOut			fo(&buf, &bufSize);

   fo.addValueCard("TESTTEST", "This is a test");
   fo << A;

   // write to file test_inmemout_float.fits
   int fd = open("test_inmem_float.fits", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
   if (fd < 0)
   {
	  cerr << "cannot open test_inmem_float.fits for writing" << endl;
      throw FitsException("");
   }

   size_t	bytes = write(fd, buf, bufSize);
   if ((size_t)bytes != bufSize)
   {
	  cerr << "cannot write test_inmem_float.fits" << endl;
      throw FitsException("");
   }
   close(fd);

   MArray<float,2> B;
   FitsIn fi("test_inmem_float.fits");
   fi >> B;

   LTL_ASSERT_( fi.getNaxis() == 2, "FITS in-memory write Naxis wrong" );
   LTL_ASSERT_( fi.getBitpix() == -32, "FITS in-memory write bitpix wrong" );
   LTL_ASSERT_( fi.getNaxis(1) == 2448, "FITS in-memory write Naxis1 wrong" );
   LTL_ASSERT_( fi.getNaxis(2) == 2050, "FITS in-memory write Naxis1 wrong" );

   LTL_ASSERT_( allof(A == B),
                "FITS in-memory data mismatch");

   LTL_ASSERT_( fi.getString( "TESTTEST" ) == "This is a test",
                "FITS AddValueCard( string ) failed" );

   fi.freeData();

   remove("test_inmem_float.fits");
}

int main(int argc, char **argv)
{
   cerr << "Testing MArray FITS I/O ..." << endl;   
   const int offtsize = sizeof(off_t);
   if(sizeof(off_t)!=8){
     cerr << "Wrong offset type off_t: Must be 64 bit, but is "
	  << offtsize * 8 << " bit."<< endl;
     throw FitsException("");
   }
   try
   {
      testfloat ();
      testdouble ();
      testbyte ();
      testshort ();
      testint ();
      testcopy ();
      testreadline ();
      testwriteregion ();
      testwriteconstarray ();
      testextensionio ();
      testinmemoryread ();
      testinmemorywrite ();
   }
   catch (exception &e){
      cerr << e.what() << endl;
      throw;
   }
}
