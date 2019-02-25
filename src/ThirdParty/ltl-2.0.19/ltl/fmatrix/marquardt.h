/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marquardt.h 522 2013-04-30 22:32:26Z cag $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#ifndef __LTL_MARQUARDT__
#define __LTL_MARQUARDT__

#define __LTL_IN_FILE_MARQUARDT__

#include <ltl/marray.h>
#include <ltl/statistics.h>
#include <ltl/fmatrix/gaussj.h>

namespace ltl {

#include <ltl/fmatrix/mrqhesse.h>
#include <ltl/fmatrix/mrqfuncs.h>

/*! \ingroup nonlinlsq
 */
//@{

//! Marquardt-Levenberg approximation of an arbitrary fitfunction TFUNC.
/*! 
  Class to fit an arbitrary function TFUNC to arbitrary data (and error).
 */
template<class TFUNC, class TPAR, class TDAT, int NPAR, int NDIM>
class Marquardt
{
   public:
      //! Construct class with fit constraints.
      inline Marquardt(const size_t initer,
                       const TPAR ainstart, const TPAR ainstep,
                       const TPAR ainmin, const TPAR ainmax,
                       const FVector<bool, NPAR> ignin = false) :
         itermax_(initer),
         alamdastart_(ainstart), alamdastep_(ainstep),
         alamdamin_(ainmin), alamdamax_(ainmax),
         have_results_(false),
         ignore_(anyof(ignin == true)), ignorePar_(ignin),
         invalid_string_("Marquardt has no valid results.")
      { }

      //! Fit to data and \f$error^2\f$ ignoring nan, start with inpar.
      inline void eval(const MArray<TDAT, NDIM>& restrict_ data,
                       const TDAT nan,
                       const MArray<TDAT, NDIM>& restrict_ error2,
                       const FVector<TPAR, NPAR>& inpar)
      {
         // set data in funtion
         function_.setData(data, nan, error2);

         // FVector<TPAR, NPAR> parameter_; // parameter is class member
         parameter_ = function_.partofit(inpar);
         FVector<TPAR, NPAR> da;
         // FMatrix<TPAR, NPAR, NPAR> covar_; // covar is class member

         // initialize matrices and vectors etc.
         TPAR alamda = alamdastart_;
         FMatrix<TPAR, NPAR, NPAR> a;
         typename FMatrix<TPAR, NPAR, NPAR>::TraceVector atrace = a.traceVector();
         FVector<TPAR, NPAR> b;
         FVector<TPAR, NPAR> partry = parameter_;
         chisquare_ = function_.marquardtCoefficients(partry,
                                                      TPAR(0),
                                                      a, b);
         TPAR old_chisquare = chisquare_; // save chi (chi^2)

         covar_ = a;
         da = b;

         i_ = size_t(0);
         while( (i_ < itermax_) && (alamda > alamdamin_) && (alamda < alamdamax_) )
         {
            // add alamda to trace of a
            const TPAR alp1 = TPAR(1.0) + alamda;
            atrace *= alp1;

            if(ignore_)
            {
               for(int i = 1; i <= NPAR; ++i)
               {
                  if(ignorePar_(i))
                  {
                     b(i) = TPAR(0);
                     a.row(i) = TPAR(0);
                     a.col(i) = TPAR(0);
                     a(i, i) = TPAR(1);
                  }
               }
            }
      
            // solve matrix equation
            //GaussJ<TPAR, NPAR>::eval(a, b);
            // add solution to parameters
            partry += GaussJ<TPAR, NPAR>::solve(a, b);
            // test function (and get new a and b)
            chisquare_ = function_.marquardtCoefficients(partry,
                                                         chisquare_,
                                                         a, b);
            // better?
            if(chisquare_ < old_chisquare)
            {
               alamda /= alamdastep_; // reduce step
               // save new matrix, vector and parameters
               parameter_ = partry;
               covar_ = a;
               da = b;
               old_chisquare = chisquare_; // save new chi^2
            }
            // worse!
            else{
               alamda *= alamdastep_; // increase step
               // restore old matrix, vector and parameters
               partry = parameter_;
               a = covar_;
               b = da;
               chisquare_ = old_chisquare; // restore old chi^2
            }
            ++i_;
         }
         chisquare_ /= TPAR(function_.getNdof()); // divide by dof
         GaussJ<TPAR, NPAR>::eval(covar_, da);
         if(i_ == itermax_)
            throw DivergenceException("Marquardt-Levenberg did not converge.");
         have_results_ = true;
         //function_.freeData();
      }

      //! Return result vector.
      inline FVector<TPAR, NPAR> getResult()
      { 
         if(have_results_)
            return function_.fittopar(parameter_, covar_.traceVector());
         throw  LinearAlgebraException(invalid_string_);
      }

      //! Return final \f$\chi^2\f$.
      inline TPAR getChiSquare() const
      {
         if(have_results_)
            return chisquare_;
         throw  LinearAlgebraException(invalid_string_);
      }

      //! Return No needed iterations.
      inline size_t getNIteration() const
      {
         if(have_results_)
            return i_;
         throw  LinearAlgebraException(invalid_string_);
      }

      //! Return \f$error^2\f$ in fit parameters.
      inline FVector<TPAR, NPAR> getVariance()
      {
         if(have_results_)
            return function_.covtoerr(covar_.traceVector(), parameter_);
         throw  LinearAlgebraException(invalid_string_);
      }


   private:
      FVector<TPAR, NPAR> parameter_;
      FMatrix<TPAR, NPAR, NPAR> covar_;
      TFUNC function_;
      const size_t itermax_;
      const TPAR alamdastart_, alamdastep_,
                 alamdamin_, alamdamax_;
      bool have_results_;
      const bool ignore_;
      const FVector<bool, NPAR> ignorePar_;
      const string invalid_string_;
      TPAR chisquare_;
      size_t i_;
};
//@}


}

#undef __LTL_IN_FILE_MARQUARDT__

#endif
