/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testsimdstatistics2.cpp 476 2010-11-12 06:00:58Z drory $
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
#define LTL_USE_SIMD
#define LTL_DEBUG_EXPRESSIONS

#include <ltl/marray.h>
#include <ltl/marray_io.h>
#include <ltl/statistics.h>

#include <iostream>


using namespace ltl;

using std::cout;
using std::endl;

const int N=57;

template<typename T> void test_count(const string&);
template<typename T> void test_allof(const string&);
template<typename T> void test_noneof(const string&);
template<typename T> void test_anyof(const string&);

int main(int argc, char **argv)
{
   cerr << "Testing MArray statistics 2 using SIMD vectorization" << endl;

   test_count<float>("float");
   test_count<double>("double");
   test_count<int>("int");
   test_count<short>("short");
   test_count<char>("char");

   test_allof<float>("float");
   test_allof<double>("double");
   test_allof<int>("int");
   test_allof<short>("short");
   test_allof<char>("char");

   test_noneof<float>("float");
   test_noneof<double>("double");
   test_noneof<int>("int");
   test_noneof<short>("short");
   test_noneof<char>("char");

   test_anyof<float>("float");
   test_anyof<double>("double");
   test_anyof<int>("int");
   test_anyof<short>("short");
   test_anyof<char>("char");
}

template<typename T>
void test_count( const string& type )
{
   cout << "Testing vectorzied count<"<<type<<">() " << endl;
   MArray<T,1> A(N);
   A = indexPosInt(A, 1);
   A(4) = 0;
   A(13) = 0;
   A(47) = 0;

   unsigned int s = count(A);
   LTL_EXPECT_(A.nelements()-3, s, "Vectorized count() failed");
}

template<typename T>
void test_allof( const string& type )
{
   cout << "Testing vectorzied allof<"<<type<<">() " << endl;
   MArray<T,1> A(N);
   A = 1;
   bool s = allof(A);
   LTL_EXPECT_(true, s, "Vectorized allof() failed");
   A(13) = 0;
   A(47) = 0;
   s = allof(A);
   LTL_EXPECT_(false, s, "Vectorized allof() failed");
}

template<typename T>
void test_noneof( const string& type )
{
   cout << "Testing vectorzied noneof<"<<type<<">() " << endl;
   MArray<T,1> A(N);
   A = 0;
   bool s = noneof(A);
   LTL_EXPECT_(true, s, "Vectorized noneof() failed");
   A(N-13) = 3;
   A(N) = 73;
   s = noneof(A);
   LTL_EXPECT_(false, s, "Vectorized noneof() failed");
}

template<typename T>
void test_anyof( const string& type )
{
   cout << "Testing vectorzied anyof<"<<type<<">() " << endl;
   MArray<T,1> A(N);
   A = 0;
   bool s = anyof(A);
   LTL_EXPECT_(false, s, "Vectorized anyof() failed");
   A(N-10) = 3;
   A(7) = 73;
   s = anyof(A);
   LTL_EXPECT_(true, s, "Vectorized anyof() failed");
}
