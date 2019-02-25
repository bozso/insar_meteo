/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: lmfit.h 547 2014-08-22 21:46:23Z landriau $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
 *                         Niv Drory <drory@mpe.mpg.de>
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

#ifndef __LTL_LMFIT__
#define __LTL_LMFIT__

#include <ltl/marray.h>
#include <ltl/fmatrix/gaussj.h>
#include <ltl/fmatrix/lusolve.h>

namespace ltl {

#include <ltl/fmatrix/mrqhesse.h>

/*! \ingroup nonlinlsq
 */
//@{
//! Marquardt-Levenberg fit to a generic function
/*!
  Class to fit an arbitrary function to data and errors by non-linear least squares using
  the Marquardt-Levenberg algorithm. Either Gauss-Jordan or LU decomposition can be used
  to solve linear systems during fitting, controlled by a template parameter.

  The function object should provide a typedef of \c value_type and \code operator() \endcode .
  Using \c TPAR to denote the type of the parameters, \c NPAR their number, and \c T the
  type of the argument:
  \code
  class function
  {
     public:
        typedef T value_type;
        value_type operator()( const T x, const FVector<TPAR,NPAR>& p, FVector<TPAR,NPAR>& df_dpi ) const
        {
           value_type value = ...;
           for( int i=1; i<=NPAR; ++i )
              df_dpi(i) = ...;
           return value;
        }
  }
  \endcode

  Here's an example class that fits a quadratic function:

  \code
   class function
   {
      public:
         typedef float value_type;
         float operator()( const float x, const FVector<float,3>& p, FVector<float,3>& df_dpi ) const
         {
            float value = p(1)*x*x + p(2)*x + p(3);
            df_dpi(1) = x*x;
            df_dpi(2) = x;
            df_dpi(3) = 1.0f;

            return value;
         }
   };
  \endcode

  This function is used in the following way:

  \code
   // just to illustrate the template parameters:
   // TFUNC is the function object,
   // TPAR the type of the parameters (needs not be the same as the input or return type of the function),
   // NPAR the number of parameters.
   // LinSolver is the solver for the linear systems during fitting. Compatible objects are
   // GaussJ (Gauss-Jordan elimination) and LUDecomposition (LU decomposition/SVD).
   // The former is faster, the latter stable against numerically singular matrices.
   template<typename TFUNC, typename TPAR, int NPAR, typename LinSolver=GaussJ<NPAR,NPAR> >
   class LMFit;

   function F;
   LMFit< function, float, 3> M( F );

   MArray<float,1> X(10);
   X = 1.27355, -0.654883, 3.7178, 2.31818, 2.6652, -2.02182, 4.82368, -4.36208, 4.84084, 2.44391;
   MArray<float,1> Y(10);
   Y = 4.30655,0.420333,18.6992,8.27967,10.8136,3.3598,29.3496,15.9234,29.4432,9.6158;
   MArray<float,1> dY2(10);
   dY2 = 1.0f;

   FVector<float,3> P0;   // initial guess.
   P0 = 0.3, -1.0, 0.0;

   M.eval( X, Y, dY2, -9999.0f, P0 );   // perform the fitting.

   cout << M.getResult() << endl;

   // the same, but using LU decomposition to invert and solve linear systems:
   LMFit< function, float, 3, LUDecomposition<float,3> > M( F );
   ...

  \endcode

  About any container can be used for passing X, Y, and dY data, as long as the container provides
  \code typedef Container::const_itertor ... \endcode ,
  \code typedef Container::value_type ... \endcode ,
  \code Container::value_type \endcode is the same as the type of the first argument to operator()
  of the function to be fit. For example, this might well be \code FVector<float,2> \endcode , and the container
  an \code MArray<FVector<float,2> > \endcode to fit a function of 2 variables (and N parameters). This
  might be a polynomial with crossterms.
 */
template<typename TFUNC, typename TPAR, int NPAR, typename Solver=GaussJ<TPAR,NPAR> >
class LMFit
{
   public:
      //! Construct class with fit constraints.
      inline LMFit( const TFUNC& func, const int itermax=1000, const TPAR tol=1e-7,
                    const FVector<bool,NPAR> ignin = false,
                    const TPAR astart=1e-3, const TPAR astep=10.0 ) :
         func_(func),
         itermax_(itermax),
         tol_(tol),
         ignorePar_(ignin),
         alamdastart_(astart), alamdastep_(astep),
         chisquare_(0),
         ignore_(anyof(ignin == true)),
         invalid_string_("Marquardt has no valid results.")
      { }

      //! specify which parameters to leave constant during fit [ignin(i)=true].
      void setIgnore( const FVector<bool,NPAR>& ignin )
      {
         ignorePar_ = ignin;
         ignore_ = anyof(ignin == true);
      }

      //! Fit to data and \f$error^2\f$ ignoring nan, start with inpar.
      template<typename TDATAX, typename TDATAY>
      void eval( const TDATAX& x, const TDATAY& y, const TDATAY& dy2,
                 const typename TDATAY::value_type nan_y,
                 const FVector<TPAR,NPAR>& inpar );

      //! Return result vector.
      inline FVector<TPAR,NPAR> getResult()
      {
         return parameter_;
      }

      //! Return final \f$\chi^2\f$.
      inline double getChiSquare() const
      {
         return chisquare_;
      }

      //! Return No needed iterations.
      inline int getNIteration() const
      {
         return i_;
      }

      //! Return diagonal of covariance matrix.
      inline FVector<TPAR,NPAR> getVariance()
      {
         FVector<TPAR,NPAR> R;
         R = covar_.traceVector();
         return R;
      }

      //! Return diagonal of covariance matrix.
      inline FMatrix<TPAR, NPAR, NPAR> getCovarianceMatrix()
      {
         return covar_;
      }

      //! Return the formal errors of the fit parameters.
      inline FVector<TPAR,NPAR> getErrors()
      {
         FVector<TPAR,NPAR> R = sqrt(fabs(covar_.traceVector())*chisquare_);

         return R;
      }

   private:
      //! Calculate actual \f$\chi^2\f$ (if better than old one) and Hessematrix.
      template<typename TDATAX, typename TDATAY>
      double marquardtCoefficients( const TDATAX& x, const TDATAY& y, const TDATAY& dy2,
                                    const typename TDATAY::value_type nan_y,
                                    const FVector<TPAR, NPAR>& parameter,
                                    const double chisquare_limit,
                                    FMatrix<TPAR, NPAR, NPAR>& a,
                                    FVector<TPAR, NPAR>& b ) const;

      const  TFUNC& func_;
      const  int itermax_;
      const  TPAR tol_;
      FVector<bool, NPAR> ignorePar_;
      const  TPAR alamdastart_, alamdastep_;
      double chisquare_;
      int    i_;
      bool ignore_;
      FVector<TPAR, NPAR> parameter_;
      FMatrix<TPAR, NPAR, NPAR> covar_;
      const  string invalid_string_;
};

//! Fit to data and \f$error^2\f$ ignoring nan, start with inpar.
template<typename TFUNC, typename TPAR, int NPAR, typename Solver>
template<typename TDATAX, typename TDATAY>
void LMFit<TFUNC, TPAR, NPAR, Solver>::eval( const TDATAX& x, const TDATAY& y, const TDATAY& dy2,
                                     const typename TDATAY::value_type nan_y,
                                     const FVector<TPAR, NPAR>& inpar )
{
   LTL_ASSERT( x.nelements() == y.nelements(), "Input x and y data not the same length!" );
   LTL_ASSERT( x.nelements() == dy2.nelements(), "Input data and errors not the same length!" );

   // copy initial parameters
   parameter_ = inpar;

   FVector<TPAR,NPAR> da;

   // initialize matrices and vectors etc.
   TPAR alamda = alamdastart_;
   FMatrix<TPAR,NPAR,NPAR> a;
   typename FMatrix<TPAR,NPAR,NPAR>::TraceVector atrace = a.traceVector();
   FVector<TPAR,NPAR> b;
   FVector<TPAR,NPAR> partry = parameter_;
   chisquare_ = marquardtCoefficients(x, y, dy2, nan_y, partry, TPAR(0), a, b);
   if (chisquare_ == 0 )
      return;

   double old_chisquare = chisquare_; // save chi (chi^2)
   double ftol = 1.1*tol_;

   int done=0;
   int ndone =4;
   covar_ = a;
   da = b;

   i_ = 0;
   do
   {
      // add alamda to trace of a
      const TPAR alp1 = TPAR(1.0) + alamda;
      atrace *= alp1;

      if (ignore_)
      {
         for (int i = 1; i <= NPAR; ++i)
         {
            if (ignorePar_(i))
            {
               b(i) = TPAR(0);
               a.row(i) = TPAR(0);
               a.col(i) = TPAR(0);
               a(i, i) = TPAR(1);
            }
         }
      }

      // solve matrix equation
      // add solution to parameters
      partry += Solver::solve(a, b);
      //partry += GaussJ<TPAR, NPAR>::solve(a, b);
      //partry += LUDecomposition<TPAR, NPAR>::solve(a, b);

      // test function (and get new a and b)
      chisquare_ = marquardtCoefficients(x, y, dy2, nan_y, partry, chisquare_, a, b);

      // better?
      if (chisquare_ < old_chisquare)
      {
         ftol = 2.0*(old_chisquare-chisquare_)/(chisquare_+old_chisquare);
         alamda /= alamdastep_; // reduce step
         // save new matrix, vector and parameters
         parameter_ = partry;
         covar_ = a;
         da = b;
         old_chisquare = chisquare_; // save new chi^2
         done = 0;
      }
      // worse!
      else
      {
         if( ::fabs( chisquare_ - old_chisquare ) < tol_ )
            ++done;
//            ftol = tol_;  // if there's no change in chi2 anymore, stop.
         alamda *= alamdastep_; // increase step
         // restore old matrix, vector and parameters
         partry = parameter_;
         a = covar_;
         b = da;
         chisquare_ = old_chisquare; // restore old chi^2
      }
      ++i_;
   } while ( (i_ < itermax_) && (ftol > tol_) && done != ndone);

   if (y.nelements() > NPAR)
   {
      chisquare_ /= double(y.nelements()-NPAR); // divide by dof
      covar_ = Solver::invert(covar_);
   }
   else
      throw LTLException("Number of DoF in Levenberg-Marquardt < 1.");
   if (i_ == itermax_)
      throw DivergenceException("Marquardt-Levenberg did not converge after max iterations.");
}


template<typename TFUNC, typename TPAR, int NPAR, typename Solver>
template<typename TDATAX, typename TDATAY>
double LMFit<TFUNC, TPAR, NPAR, Solver>::marquardtCoefficients( const TDATAX& x, const TDATAY& y, const TDATAY& dy2,
                                                      const typename TDATAY::value_type nan_y,
                                                      const FVector<TPAR,NPAR>& parameter,
                                                      const double chisquare_limit,
                                                      FMatrix<TPAR,NPAR,NPAR>& a,
                                                      FVector<TPAR,NPAR>& b ) const
{
   // TPAR funcconst1(parameter(1), ..., paramater(NPAR));
   // ...
   // TPAR funcconstNDIM(parameter(1), ..., parameter(NPAR));
   // clear output array and vector
   tMatClearHesse<TPAR,NPAR>::eval(a);
   b = TPAR(0);
   // vector for derivative
   FVector<TPAR,NPAR> derivative;

   // loop over data
   typename TDATAX::const_iterator x_i = x.begin();
   typename TDATAY::const_iterator y_i = y.begin();
   typename TDATAY::const_iterator dy_i = dy2.begin();

   double chisquare = 0;

   while(!y_i.done())
   {
      const typename TDATAX::value_type tmp_x = *x_i;
      const typename TDATAY::value_type tmp_y = *y_i;
      const typename TDATAY::value_type tmp_dy2 = *dy_i;

      if( (tmp_y != nan_y) && (tmp_dy2 > TPAR(0)) )
      {
         const typename TFUNC::value_type fun = func_( tmp_x, parameter, derivative );
         const typename TFUNC::value_type df = tmp_y - fun;
         chisquare += double(df*df) / double(tmp_dy2);
         if( (chisquare > chisquare_limit) && (chisquare_limit != 0) )
            return chisquare;
         tMatHesse<TPAR,NPAR>::eval(derivative, df, tmp_dy2, a, b);
      }
      ++y_i;
      ++x_i;
      ++dy_i;
   }
   // fill up rest of matrix
   tMatFillHesse<TPAR,NPAR>::eval(a);
   return chisquare;
}

//@}

}

#endif  // __LTL_LMFIT__
