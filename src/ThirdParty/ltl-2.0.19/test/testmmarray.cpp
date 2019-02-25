/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testmmarray.cpp 476 2010-11-12 06:00:58Z drory $
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
#include <string>

using namespace ltl;

using std::cout;
using std::endl;


int main(int argc, char **argv)
{
   cerr << "Testing mapped MArray basics ..." << endl;
   
   MArray<float,2> A(3,3,true);
   A = 42.;
   LTL_ASSERT_( A(1,1)==42. && A(3,3)==42.,
                "MArray init failed" );

   A.setBase(-1,-1);
   LTL_ASSERT_( A(-1,-1)==42. && A(1,1)==42.,
                "MArray setBase() failed" );

   A = A * 2.;
   A = A - A/2.;   
   LTL_ASSERT_( allof(A==42.),
                "MArray basic expressions failed" );
   
   A(-1,-1) = 0.;
   LTL_ASSERT_( !allof(A) && anyof(A) && !noneof(A),
                "MArray logic expressions failed" );
   
   MArray<float,1> B = A(-1,Range::all());
   B = 1;
   LTL_ASSERT_( allof(B==1) && allof( A(-1,Range::all())==1 ),
                "MArray slice failed" );

   MArray<float,1> C = A(Range::all(),1);
   C = 2;
   LTL_ASSERT_( allof(C==2) && allof( A(Range::all(),1)==2 ),
                "MArray slice failed" );
   
   A = 10.0;
   A = log10(A);
   LTL_ASSERT_( allof(A==1.0),
                "MArray basic mathlib expressions failed" );

   A = 0.0;
   A(0,0)  = 1;
   A(-1,0) = 3;
   MArray<float,2>::IndexSet l = where( A!=0.0 );
   LTL_ASSERT_( l.size() == 2, 
                "MArray where() failed" );
   
   A[l] = 5;
   LTL_ASSERT_( A(0,0) == 5 && A(-1,0) == 5, 
                "MArray where()/operator[] failed" );
   A[l] = 0;
   LTL_ASSERT_( noneof(A),
                "MArray where()/operator[] failed" );
   
   
   A(-1,Range::all()) = 2;
   A = merge( A, 1./A, -1. );
   LTL_ASSERT_( allof( A(-1,Range::all()) == 0.5 ),
                "MArray merge() failed" );
   
   A(-1,Range::all()) = -1;
   LTL_ASSERT_( allof( A == -1 ),
                "MArray merge() failed" );
   
   MArray<float,3> AA(10,10,10,true);
   AA = indexPosFlt(AA,1) + 10*indexPosFlt(AA,2) + 100*indexPosFlt(AA,3);
   LTL_ASSERT_( AA(1,1,1)==111 && AA(6,9,2)==296 && AA(9,9,9)==999 ,
                "MArray indexPos() failed" );
   
   MArray<float,2> BB = AA(Range(2,7,2),5,Range(3,8,3));
   MArray<float,2> CC(3,2,true);
   CC = 352,354,356,
        652,654,656;
   LTL_ASSERT_( allof(BB==CC),
                "MArray subarray failed" );
   
   MArray<float,1> DD = BB(2,Range::all());
   MArray<float,1> EE(2,true);
   EE = 354,654;   
   LTL_ASSERT_( allof(DD==EE),
                "MArray complex slice failed" );
   
   DD = 555;
   LTL_ASSERT_( allof(DD==555),
                "MArray complex slice assign failed" );
   LTL_ASSERT_( AA(4,5,3)==555 && AA(4,5,6)==555,
                "MArray complex slice assign failed" );
   
   {      
      MArray<float,3> MAP(AA.shape(), true, "test.map");
      MAP = AA;
   }
   {
      int dims[3] = {10,10,10};
      MArray<float,3> MAP(string("test.map"), dims);
      LTL_ASSERT_( allof(MAP==AA),
                   "MArray Map read failed." );
   }

}
