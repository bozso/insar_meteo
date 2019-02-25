/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testascio.cpp 383 2009-06-18 10:22:27Z snigula $
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
#include <ltl/ascio.h>
#include <ltl/statistics.h>
#include <ltl/marray_io.h>
#include <iostream>
#include <fstream>

using namespace ltl;

using std::cout;
using std::endl;

int main(int argc, char **argv);

void testint( void )
{

   MArray<int,1> res1(7);
   res1 = 3, 4, 5, 6, 7, 8, 9;
   MArray<int,1> res1l1a(3);
   res1l1a = 3, 4, 5;
   MArray<int,1> res1l1b(2);
   res1l1b = 6, 7;
   MArray<int,1> res1l1c(2);
   res1l1c = 8, 9;
   MArray<int,1> res1l2a(4);
   res1l2a = 3, 4, 5, 6;
   MArray<int,1> res1l2b(5);
   res1l2b = 5, 6, 7, 8, 9;
   MArray<int,2> res2(3,7);
   res2 = 
      3, 4, 5, 
      4, 5, 6, 
      5, 6, 7, 
      6, 7, 8,
      7, 8, 9,
      8, 9, 0,
      9, 0, 1;
   MArray<int,2> res2l1a(3,3);
   res2l1a = 
      3, 4, 5, 
      4, 5, 6, 
      5, 6, 7;
   MArray<int,2> res2l1b(3,2);
   res2l1b = 
      6, 7, 8,
      7, 8, 9;
   MArray<int,2> res2l1c(3,2);
   res2l1c = 
      8, 9, 0,
      9, 0, 1;
   MArray<int,2> res2l2a(3,4);
   res2l2a = 
      3, 4, 5, 
      4, 5, 6, 
      5, 6, 7, 
      6, 7, 8;
   MArray<int,2> res2l2b(3,5);
   res2l2b = 
      5, 6, 7, 
      6, 7, 8,
      7, 8, 9,
      8, 9, 0,
      9, 0, 1;

   cerr << endl << "Testing integer data file" << endl;
   cerr << "Creating testfile...";
   std::ofstream file("test.dat");
   
   file << "1 2 3 4 5" << endl
        << " # ugly comment" << endl
        << "2 3 4 5 6" << endl
        << "  3  4 5 6 7" << endl
        << "4 5 6 07 8" << endl
        << "#comment" << endl
        << "5   6 7 8 9" << endl
        << "6 7 8 9 0" << endl
        << "7  8 9 0 1" << endl;
   file.close();

   cerr << "done" << endl;

   AscFile f( "test.dat" );
   
   cerr << "Testing rows()...";

   if( f.rows() != 7 ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing cols()...";

   if( f.cols() != 5 ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumn()...";

   MArray<int,1> read1 = f.readIntColumn(3);
   
   if( ! allof( read1 == res1 ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumns()...";

   MArray<int,2> read2 = f.readIntColumns(3,5);
   
   if( ! allof( read2 == res2 ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;


   cerr << "Testing readIntColumn() with lineranges...";

   MArray<int,1> read1l1a = f.readIntColumn(3,1,3);

   if( ! allof( read1l1a == res1l1a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumn() with lineranges...";

   MArray<int,1> read1l1b = f.readIntColumn(3,4,2);
   
   if( ! allof( read1l1b == res1l1b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumn() with lineranges...";

   MArray<int,1> read1l1c = f.readIntColumn(3,6,2);
   
   if( ! allof( read1l1c == res1l1c ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumn() with lineranges...";

   MArray<int,1> read1l2a = f.readIntColumn(3,1,4);
   
   if( ! allof( read1l2a == res1l2a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumn() with lineranges...";

   MArray<int,1> read1l2b = f.readIntColumn(3,3,5);
   
   if( ! allof( read1l2b == res1l2b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;



   cerr << "Testing readIntColumns() with lineranges...";

   MArray<int,2> read2l1a = f.readIntColumns(3,5,1,3);
   
   if( ! allof( read2l1a == res2l1a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumns() with lineranges...";

   MArray<int,2> read2l1b = f.readIntColumns(3,5,4,2);
     
   if( ! allof( read2l1b == res2l1b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumns() with lineranges...";

   MArray<int,2> read2l1c = f.readIntColumns(3,5,6,2);
   
   if( ! allof( read2l1c == res2l1c ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumns() with lineranges...";

   MArray<int,2> read2l2a = f.readIntColumns(3,5,1,4);
   
   if( ! allof( read2l2a == res2l2a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readIntColumns() with lineranges...";

   MArray<int,2> read2l2b = f.readIntColumns(3,5,3,5);
   
   if( ! allof( read2l2b == res2l2b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

}
      
void testfloat( void )
{

   MArray<float,1> res1(7);
   res1 = 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8;
   MArray<float,1> res1l1a(3);
   res1l1a = 3.2, 4.3, 5.4;
   MArray<float,1> res1l1b(2);
   res1l1b = 6.5, 7.6;
   MArray<float,1> res1l1c(2);
   res1l1c = 8.7, 9.8;
   MArray<float,1> res1l2a(4);
   res1l2a = 3.2, 4.3, 5.4, 6.5;
   MArray<float,1> res1l2b(5);
   res1l2b = 5.4, 6.5, 7.6, 8.7, 9.8;
   MArray<float,2> res2(3,7);
   res2 = 
      3.2, 4.3, 5.4, 
      4.3, 5.4, 6.5, 
      5.4, 6.5, 7.6, 
      6.5, 7.6, 8.7,
      7.6, 8.7, 9.8,
      8.7, 9.8, 0.9,
      9.8, 0.9, 1.0;
   MArray<float,2> res2l1a(3,3);
   res2l1a = 
      3.2, 4.3, 5.4, 
      4.3, 5.4, 6.5, 
      5.4, 6.5, 7.6;
   MArray<float,2> res2l1b(3,2);
   res2l1b = 
      6.5, 7.6, 8.7,
      7.6, 8.7, 9.8;
   MArray<float,2> res2l1c(3,2);
   res2l1c = 
      8.7, 9.8, 0.9,
      9.8, 0.9, 1.0;
   MArray<float,2> res2l2a(3,4);
   res2l2a = 
      3.2, 4.3, 5.4, 
      4.3, 5.4, 6.5, 
      5.4, 6.5, 7.6, 
      6.5, 7.6, 8.7;
   MArray<float,2> res2l2b(3,5);
   res2l2b = 
      5.4, 6.5, 7.6, 
      6.5, 7.6, 8.7,
      7.6, 8.7, 9.8,
      8.7, 9.8, 0.9,
      9.8, 0.9, 1.0;

   cerr << endl << "Testing float data file" << endl;
   cerr << "Creating testfile...";
   std::ofstream file("test.dat");
   
   file << "1.0 2.1 3.20 4.3 5.4" << endl
        << " # ugly comment" << endl
        << "2.100000 3.2 4.3 5.40 6.5" << endl
        << "  3.200  04.30 5.4 6.5 7.6" << endl
        << "4.3 5.4 6.50 7.6 8.7" << endl
        << "#comment" << endl
        << "5.40000000000000000000000000000   6.5 7.60 8.70 9.8" << endl
        << "6.5 7.6 8.7 9.8 0.9000000000000000000000000000000000000000" << endl
        << "7.6  8.7000 9.80 0.90 1.00" << endl;
   file.close();

   cerr << "done" << endl;

   AscFile f( "test.dat" );

   cerr << "Testing rows()...";

   if( f.rows() != 7 ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing cols()...";

   if( f.cols() != 5 ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumn()...";

   MArray<float,1> read1 = f.readFloatColumn(3);
   if( ! allof( read1 == res1 ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumns()...";

   MArray<float,2> read2 = f.readFloatColumns(3,5);
   
   if( ! allof( read2 == res2 ) ) {
	cerr << read2 << endl;
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;


   cerr << "Testing readFloatColumn() with lineranges...";

   MArray<float,1> read1l1a = f.readFloatColumn(3,1,3);

   if( ! allof( read1l1a == res1l1a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumn() with lineranges...";

   MArray<float,1> read1l1b = f.readFloatColumn(3,4,2);
   
   if( ! allof( read1l1b == res1l1b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumn() with lineranges...";

   MArray<float,1> read1l1c = f.readFloatColumn(3,6,2);
   
   if( ! allof( read1l1c == res1l1c ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumn() with lineranges...";

   MArray<float,1> read1l2a = f.readFloatColumn(3,1,4);
   
   if( ! allof( read1l2a == res1l2a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumn() with lineranges...";

   MArray<float,1> read1l2b = f.readFloatColumn(3,3,5);
   
   if( ! allof( read1l2b == res1l2b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;



   cerr << "Testing readFloatColumns() with lineranges...";

   MArray<float,2> read2l1a = f.readFloatColumns(3,5,1,3);
   
   if( ! allof( read2l1a == res2l1a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumns() with lineranges...";

   MArray<float,2> read2l1b = f.readFloatColumns(3,5,4,2);
     
   if( ! allof( read2l1b == res2l1b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumns() with lineranges...";

   MArray<float,2> read2l1c = f.readFloatColumns(3,5,6,2);
   
   if( ! allof( read2l1c == res2l1c ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumns() with lineranges...";

   MArray<float,2> read2l2a = f.readFloatColumns(3,5,1,4);
   
   if( ! allof( read2l2a == res2l2a ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;

   cerr << "Testing readFloatColumns() with lineranges...";

   MArray<float,2> read2l2b = f.readFloatColumns(3,5,3,5);
   
   if( ! allof( read2l2b == res2l2b ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;




   MArray<float,2> res3(2,7);
   res3 = 
      3.2, 5.4, 
      4.3, 6.5, 
      5.4, 7.6, 
      6.5, 8.7,
      7.6, 9.8,
      8.7, 0.9,
      9.8, 1.0;

   int c[2] = {3,5};
   MArray<float,2> read3 = f.readFloatColumns(c,2);
   
   if( ! allof( read3 == res3 ) ) {
      cerr << "failed" << endl;
      exit(-1);
   } else 
      cerr << "passed" << endl;
}
      
void testheader( void )
{

   cerr << endl << "Testing getHeader with normal comments" << endl;
   cerr << "Creating testfile...";
   std::ofstream file("test.dat");
   
   file << "# Comment line 1 " << endl
        << "  # Comment line 2 " << endl
        << "# Comment line 3 " << endl
        << "   " << endl
        << "# Comment line 4 " << endl
        << "" << endl
        << "        # Comment line 5 " << endl
        << "# Comment line 6 " << endl
        << "1.0 2.1 3.20 4.3 5.4" << endl
        << " # ugly comment" << endl
        << "2.100000 3.2 4.3 5.40 6.5" << endl
        << "  3.200  04.30 5.4 6.5 7.6" << endl
        << "4.3 5.4 6.50 7.6 8.7" << endl
        << "#comment" << endl
        << "5.40000000000000000000000000000   6.5 7.60 8.70 9.8" << endl
        << "6.5 7.6 8.7 9.8 0.9000000000000000000000000000000000000000" << endl
        << "7.6  8.7000 9.80 0.90 1.00" << endl;
   file.close();

   cerr << "done" << endl;

   AscFile f( "test.dat" );
   
   cerr << "Testing with skipped comment signs...";

   vector<string> v1;

   f.getHeader( v1, false );
   
   if( v1[0] != " Comment line 1 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 1 \" got \"" << v1[0] << "\"" << endl;
   else if( v1[1] != " Comment line 2 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 2 \" got \"" << v1[1] << "\"" << endl;
   else if( v1[2] != " Comment line 3 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 3 \" got \"" << v1[2] << "\"" << endl;
   else if( v1[3] != " Comment line 4 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 4 \" got \"" << v1[3] << "\"" << endl;
   else if( v1[4] != " Comment line 5 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 5 \" got \"" << v1[4] << "\"" << endl;
   else if( v1[5] != " Comment line 6 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 6 \" got \"" << v1[5] << "\"" << endl;
   else
      cerr << "passed" << endl;


   cerr << "Testing without skipped comment signs...";

   v1.clear();

   f.getHeader( v1, true );

   if( v1[0] != "# Comment line 1 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 1 \" got \"" << v1[0] << "\"" << endl;
   else if( v1[1] != "# Comment line 2 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 2 \" got \"" << v1[1] << "\"" << endl;
   else if( v1[2] != "# Comment line 3 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 3 \" got \"" << v1[2] << "\"" << endl;
   else if( v1[3] != "# Comment line 4 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 4 \" got \"" << v1[3] << "\"" << endl;
   else if( v1[4] != "# Comment line 5 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 5 \" got \"" << v1[4] << "\"" << endl;
   else if( v1[5] != "# Comment line 6 " )
      cerr << "failed" << endl 
           << "Expected \"# Comment line 6 \" got \"" << v1[5] << "\"" << endl;
   else
      cerr << "passed" << endl;

   
   cerr << endl << "Testing getHeader with weird comments" << endl;
   cerr << "Creating testfile...";
   std::ofstream file2("test.dat");
   
   file2 << "COMMENT Comment line 1 " << endl
        << "  COMMENT Comment line 2 " << endl
        << "COMMENT Comment line 3 " << endl
        << "   " << endl
        << "COMMENT Comment line 4 " << endl
        << "" << endl
        << "        COMMENT Comment line 5 " << endl
        << "COMMENT Comment line 6 " << endl
        << "1.0 2.1 3.20 4.3 5.4" << endl
        << " COMMENT ugly comment" << endl
        << "2.100000 3.2 4.3 5.40 6.5" << endl
        << "  3.200  04.30 5.4 6.5 7.6" << endl
        << "4.3 5.4 6.50 7.6 8.7" << endl
        << "COMMENTcomment" << endl
        << "5.40000000000000000000000000000   6.5 7.60 8.70 9.8" << endl
        << "6.5 7.6 8.7 9.8 0.9000000000000000000000000000000000000000" << endl
        << "7.6  8.7000 9.80 0.90 1.00" << endl;
   file.close();

   cerr << "done" << endl;

   AscFile f2( "test.dat", 0, "COMMENT" );
   
   cerr << "Testing with skipped comment signs...";

   v1.clear();

   f2.getHeader( v1, false );

   if( v1[0] != " Comment line 1 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 1 \" got \"" << v1[0] << "\"" << endl;
   else if( v1[1] != " Comment line 2 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 2 \" got \"" << v1[1] << "\"" << endl;
   else if( v1[2] != " Comment line 3 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 3 \" got \"" << v1[2] << "\"" << endl;
   else if( v1[3] != " Comment line 4 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 4 \" got \"" << v1[3] << "\"" << endl;
   else if( v1[4] != " Comment line 5 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 5 \" got \"" << v1[4] << "\"" << endl;
   else if( v1[5] != " Comment line 6 " )
      cerr << "failed" << endl 
           << "Expected \" Comment line 6 \" got \"" << v1[5] << "\"" << endl;
   else
      cerr << "passed" << endl;


   cerr << "Testing without skipped comment signs...";

   v1.clear();

   f2.getHeader( v1, true );

   if( v1[0] != "COMMENT Comment line 1 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 1 \" got \"" << v1[0] << "\"" << endl;
   else if( v1[1] != "COMMENT Comment line 2 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 2 \" got \"" << v1[1] << "\"" << endl;
   else if( v1[2] != "COMMENT Comment line 3 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 3 \" got \"" << v1[2] << "\"" << endl;
   else if( v1[3] != "COMMENT Comment line 4 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 4 \" got \"" << v1[3] << "\"" << endl;
   else if( v1[4] != "COMMENT Comment line 5 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 5 \" got \"" << v1[4] << "\"" << endl;
   else if( v1[5] != "COMMENT Comment line 6 " )
      cerr << "failed" << endl 
           << "Expected \"COMMENT Comment line 6 \" got \"" << v1[5] << "\"" << endl;
   else
      cerr << "passed" << endl;

}
      

int main(int argc, char **argv)
{
   cerr << "Testing MArray ASCII I/O ..." << endl;   
   testint();
   testfloat();
   testheader();
}
