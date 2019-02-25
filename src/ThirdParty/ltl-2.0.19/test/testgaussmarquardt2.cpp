/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testgaussmarquardt2.cpp 476 2010-11-12 06:00:58Z drory $
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

#include <ltl/nonlinlsqfit.h>
#include <ltl/marray_io.h>

#include <iostream>

using namespace ltl;

using std::cout;
using std::cerr;
using std::endl;

template<class FUNCTION, class TPAR, class TDAT, int NPAR, int NDAT>
void testmarquardt(FVector<TPAR, NPAR>& startpar,
                   FVector<bool, NPAR>& ignore,
                   MArray<TDAT, NDAT>& A,
                   MArray<TDAT, NDAT>& A_error)
{
   Marquardt< FUNCTION,
      TPAR, TDAT, NPAR, NDAT >
      gaussmrq(160, 1.0e-5, 12.0,
               1.0e-10, 1.0e10,
               ignore);

   gaussmrq.eval(A, 0.0f, A_error, startpar);
   FVector<TPAR, NPAR> parameter;
   parameter = gaussmrq.getResult();
   cerr << endl;
   cout << "Chi^2 = " << gaussmrq.getChiSquare()
        << " after " << gaussmrq.getNIteration()
        << " iterations" << endl;
   cout << "Solution: " << parameter << endl;
   cout << "Error^2: " << gaussmrq.getVariance() << endl;
}


int main(int argc, char **argv)
{
   try
   {
      cerr << "Testing nonlinear least squares fit:" << endl;

      FVector<double, 7> inputv;
      inputv =
         10000.0, 10000.0, 1.0, 2.0, 25.0, 6.0, 6.0;
      
      MArray<float, 2> A(11,11);
      
      const double cw=cos(inputv(5) * M_PI / 180.0);
      const double sw=sin(inputv(5) * M_PI / 180.0);

#ifndef __xlC__
      A = inputv(1) +
         inputv(2) * exp( -( (4.0 * M_LN2 / pow2(inputv(3))) *
                             pow2((indexPosFlt(A,1) - inputv(6)) * cw +
                                  (indexPosFlt(A,2) - inputv(7)) * sw) +
                             (4.0 * M_LN2 / pow2(inputv(4))) *
                             pow2((indexPosFlt(A,2) - inputv(7)) * cw -
                                  (indexPosFlt(A,1) - inputv(6)) * sw) ) );
#else
      const double aw = 4.0*M_LN2/pow2(inputv(3));
      const double bw = 4.0*M_LN2/pow2(inputv(4));
      A =  aw*pow2((indexPosFlt(A,1) - inputv(6)) * cw +
                   (indexPosFlt(A,2) - inputv(7)) * sw );
      A += bw*pow2((indexPosFlt(A,2) - inputv(7)) * cw -
                   (indexPosFlt(A,1) - inputv(6)) * sw );

      A =  inputv(1) + inputv(2)*exp(-A);
#endif

      MArray<float, 2> A_error(A.shape());
      A_error = sqrt(A);

      cout << "   Input parameters: "
           << inputv << endl;

      cout << "   => Input Matrix: "
           << A << endl << endl;
      
      {
         cerr << "   Marquardt-Levenberg fitting 7-parameter Gaussian ..." << endl
              << "   Angular version: C + A * exp(x_width, y_width," << endl
              << "                                angle, x_0, y_0)" << endl;
         FVector<bool, 7> ignore = false;
         //ignore (7) = true;
         FVector<double, 7> startpar;
         startpar = 9990,11000.0, 1.2, 1.8, 20.0, 5.9, 6.1;         
         testmarquardt< Gaussian<double, float, 7, 2>,
            double, float, 7, 2 >(startpar, ignore, A, A_error);
      }
      {
         cerr << "   Marquardt-Levenberg fitting 5-parameter Gaussian ..." << endl
              << "   Angular version: C + A * exp(x_width, y_width," << endl
              << "                                angle)" << endl;
         FVector<bool, 5> ignore = false;
         //ignore (5) = true;
         FVector<double, 5> startpar;
         startpar = 9990,11000.0, 1.2, 1.8, 20.0;         
         testmarquardt< Gaussian<double, float, 5, 2>,
            double, float, 5, 2 >(startpar, ignore, A, A_error);
      }
      {
         cerr << "   Marquardt-Levenberg fitting 3-parameter Gaussian ..." << endl
              << "   Angular version: C + A * exp(width)" << endl;
         FVector<bool, 3> ignore = false;
         //ignore (3) = true;
         FVector<double, 3> startpar;
         startpar = 9999,11000.0, 1.5;
         testmarquardt< Gaussian<double, float, 3, 2>,
            double, float, 3, 2 >(startpar, ignore, A, A_error);
      }
      {
         cerr << "   Marquardt-Levenberg fitting 7-parameter Gaussian ..." << endl
              << "   Polynomial version: C + A * exp(x_width, y_width," << endl
              << "                                   xy_width, x_0, y_0)" << endl;
         FVector<bool, 7> ignore = false;
         //ignore (7) = true;
         FVector<double, 7> startpar;
         startpar = 9990,11000.0, 1.2, 1.8, 20.0, 5.9, 6.1;         
         testmarquardt< PolyGaussian<double, float, 7, 2>,
            double, float, 7, 2 >(startpar, ignore, A, A_error);
      }
      {
         cerr << "   Marquardt-Levenberg fitting 5-parameter Gaussian ..." << endl
              << "   Polynomial version: C + A * exp(x_width, y_width," << endl
              << "                                   xy_width)" << endl;

         FVector<bool, 5> ignore = false;
         //ignore (5) = true;
         FVector<double, 5> startpar;
         startpar = 9990,11000.0, 1.2, 1.8, 20.0;         
         testmarquardt< PolyGaussian<double, float, 5, 2>,
            double, float, 5, 2 >(startpar, ignore, A, A_error);
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
