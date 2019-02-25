/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testpolynomfit2.cpp 363 2008-07-10 15:09:44Z drory $
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
//#define LTL_DEBUG_EXPRESSIONS

#ifdef LTL_TEMPLATE_LOOP_LIMIT
#undef LTL_TEMPLATE_LOOP_LIMIT
#endif
#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/marray.h>
#include <ltl/linlsqfit.h>
#include <ltl/marray_io.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::cerr;
using std::endl;

template<class TPAR, class TDAT, int ORDER, bool EXT, int NDAT>
void testpolynom(MArray<TDAT, NDAT>& A,
                 MArray<TDAT, NDAT>& A_error)
{
   string comment;
   MArray<TDAT, NDAT> polfit =
      PolynomFit<TPAR, TDAT, ORDER, EXT, NDAT>::fit(A, A_error, comment);
   cout << comment << endl;
}


int main(int argc, char **argv)
{
   try
   {
      cerr << "Testing linear least squares fit:" << endl;

      FVector<double, 7> inputv;
      inputv =
         10000.0, 10000.0, 1.0, 2.0, 25.0, 6.0, 6.0;
      
      MArray<float, 2> A(11,11);
      
      const double cw=cos(inputv(5) * M_PI / 180.0);
      const double sw=sin(inputv(5) * M_PI / 180.0);

#ifndef __xlC__
      A = inputv(1) +
         inputv(2) * exp( -( (4.0 * M_LN2 / pow2(inputv(3))) *
                             pow2((indexPosDbl(A,1) - inputv(6)) * cw +
                                  (indexPosDbl(A,2) - inputv(7)) * sw) +
                             (4.0 * M_LN2 / pow2(inputv(4))) *
                             pow2((indexPosDbl(A,2) - inputv(7)) * cw -
                                  (indexPosDbl(A,1) - inputv(6)) * sw) ) );
#else
      const double aw = 4.0*M_LN2/pow2(inputv(3));
      const double bw = 4.0*M_LN2/pow2(inputv(4));
      A =  aw*pow2((indexPosDbl(A,1) - inputv(6)) * cw +
                   (indexPosDbl(A,2) - inputv(7)) * sw );
      A += bw*pow2((indexPosDbl(A,2) - inputv(7)) * cw -
                   (indexPosDbl(A,1) - inputv(6)) * sw );

      A =  inputv(1) + inputv(2)*exp(-A);
#endif

      MArray<float, 2> A_error(A.shape());
      A_error = sqrt(A);

      cout << "   Input parameters: "
           << inputv << endl;

      cout << "   => Input Matrix: "
           << A << endl << endl;
      
      MArray<float, 1> B( A(6 ,Range::all()) );
      MArray<float, 1> B_error( A_error(6 ,Range::all()) );
         
      cout << "   => Input Slice (Matrix Col 6): "
           << B << endl << endl;
      
      {
         cerr << "   testing 1D - order 2, simple" << endl;
         testpolynom<double, float, 2, false, 1>(B, B_error);
      }
      {
         cerr << "   testing 1D - order 4, simple" << endl;
         testpolynom<double, float, 4, false, 1>(B, B_error);
      }
      {
         cerr << "   testing 2D - order 1, extended" << endl;
         testpolynom<double, float, 1, true, 2>(A, A_error);
      }
      {
         cerr << "   testing 2D - order 3, simple" << endl;
         testpolynom<double, float, 3, false, 2>(A, A_error);
      }
   }
//    catch(ltl::LinearAlgebraException e)
//    {
//       cerr << e.what() << endl
//            << "May be OK nevertheless... ";
//    }
   catch(std::exception& e)
   {
      cerr << e.what() << endl;
      return -1;
   }
   return 0;   
}
