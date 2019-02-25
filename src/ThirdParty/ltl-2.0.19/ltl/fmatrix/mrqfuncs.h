/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: mrqfuncs.h 561 2015-04-30 12:58:18Z drory $
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

#ifndef __LTL_MRQFUNCS__
#define __LTL_MRQFUNCS__

#ifndef __LTL_IN_FILE_MARQUARDT__
#error "<ltl/fmatrix/mrqfuncs.h> must be included via <ltl/fmatrix/marquardt.h>, never alone!"
#endif

#include <cfloat>

/*! \ingroup nonlinlsq
 */
//@{
//! Mother (and example) for function class suitable for ltl::Marquardt.
template<class TPAR, class TDAT, int NPAR, int NDIM>
class MRQFunction
{
   public:

      //inline MRQFunction() { };

      //! Set data, error and NaN value.
      inline void setData(const MArray<TDAT, NDIM>& indata,
                          const TDAT in_nan,
                          const MArray<TDAT, NDIM>& inerror2)
      {
         ndof_ = count(inerror2 > TPAR(0.0)) - NPAR;
         if(ndof_ <= 0)
            throw LinearAlgebraException("No. valid data points <= No. parameters");
         error2_.makeReference(inerror2);
         data_.makeReference(indata);
         nan_data_ = in_nan;
      }
      
      //! Free data and error
      inline void freeData()
      {
         data_.free();
         error2_.free();
      }
      
      //! Convert external fit parameters to internal representation.
      inline static FVector<TPAR, NPAR>
      partofit(const FVector<TPAR, NPAR>& parameter)
      {
         FVector<TPAR, NPAR> fitpar;
         // fitpar(1) = f(parameter(1), ..., parameter(NPAR));
         // ...
         // fitpar(NPAR) = f(parameter(1), ..., parameter(NPAR));
         fitpar = parameter;
         return fitpar;
      }
      
      //! Convert internal fit parameters to external representation.
      inline static FVector<TPAR, NPAR>
      fittopar(const FVector<TPAR, NPAR>& fitpar,
               const typename FMatrix<TPAR, NPAR, NPAR>::TraceVector& trace)
      {
         FVector<TPAR, NPAR> parameter;
         // parameter(1) = f(fitpar(1), ..., fitpar(NPAR));
         // ...
         // parameter(NPAR) = f(fitpar(1), ..., fitpar(NPAR));
         parameter = fitpar;
         return parameter;
      }
      
      //! Calculate external error in parameters from internal covariance matrix.
      inline static FVector<TPAR, NPAR>
      covtoerr(const typename FMatrix<TPAR, NPAR, NPAR>::TraceVector& trace,
               const FVector<TPAR, NPAR>& fitpar)
      {
         FVector<TPAR, NPAR> error;
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         // error(1) = f(trace(1), ..., trace(NPAR), fitpar(1), ..., fitpar(NPAR));
         // ...
         // error(NPAR) = f(trace(1), ..., trace(NPAR), fitpar(1), ..., fitpar(NPAR));
         error = trace;
         return error;
      }
      
      //! Calculate actual \f$\chi^2\f$ (if better than old one) and Hessematrix.
      inline TPAR marquardtCoefficients(const FVector<TPAR, NPAR>& restrict_ parameter,
                                        const TPAR chisquare_limit,
                                        FMatrix<TPAR, NPAR, NPAR>& restrict_ a,
                                        FVector<TPAR, NPAR>& restrict_ b) const
      {
         // TPAR funcconst1(parameter(1), ..., paramater(NPAR));
         // ...
         // TPAR funcconstNDIM(parameter(1), ..., parameter(NPAR));
         // clear output array and vector
         tMatClearHesse<TPAR, NPAR>::eval(a);
         b = TPAR(0);
         // vector for derivative
         FVector<TPAR, NPAR> derivative;

         // loop over data
         typename MArray<TDAT, NDIM>::const_iterator data_i = data_.begin();
         typename MArray<TDAT, NDIM>::IndexIterator index_i = data_.indexBegin();
         typename MArray<TDAT, NDIM>::const_iterator error2_i = error2_.begin();

         TPAR chisquare = TPAR(0);

         while(!data_i.done())
         {
            if( (*data_i != nan_data_) && (*error2_i > TPAR(0)) )
            {
               // calculate derivative
               // derivative(1) =  f(parameter(1), ..., parameter(NPAR),
               //                    index_i(1), ..., index_i(NDIM));
               // ...
               // derivative(NPAR) =  f(parameter(1), ..., parameter(NPAR),
               //                       index_i(1), ..., index_i(NDIM));
               // calculate value of function at coordinates
               // const TPAR fun =  f(parameter(1), ..., parameter(NPAR),
               //                     index_i(1), ..., index_i(NDIM));

               // const TPAR df = TPAR(*data_i - TDAT(fun));
               // const TPAR sig2 = 1.0;
               // const TPAR sig2 = *error2_i;
               // chisquare += df * df / sig2;
               if( (chisquare > chisquare_limit) &&
                   (chisquare_limit != TPAR(0)) )
                  return chisquare;
               // tMatHesse<TPAR, NPAR>::eval(derivative, df, sig2, a, b);
            }
            ++data_i;
            ++index_i;
            ++error2_i;
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, NPAR>::eval(a);
         return chisquare;
      }

      //! Return degrees of freedom for fit.
      inline size_t getNdof() const
      { return size_t(ndof_); }

   protected:
      MArray<TDAT, NDIM> data_;
      TDAT nan_data_;
      MArray<TDAT, NDIM> error2_;
      int ndof_;
};

//! Approximation of a Gaussian function via Marquardt-Levenberg algorithm.
template<class TPAR, class TDAT, int NPAR, int NDIM>
class Gaussian : public MRQFunction<TPAR, TDAT, NPAR, NDIM>
{ };

// angular version

// 7 parameter Gaussian
// f(x, y) = C + A * exp( -( (4 ln(2) / x_w^2) *
//                           ( (x - x_0) * cos(b) + (y - y_0) * sin(b) )^2 +
//                           (4 ln(2) / y_w^2) *
//                           ( (y - y_0) * cos(b) - (x - x_0) * sin(b) )^2 ) )
// -> i.e. angular version
// f(x, y) = a1 +
//           a2 * exp( -(a3 * ((x - a6) * cos(a5) + (y - a7) * sin(a5))^2 +
//                       a4 * ((y - a7) * cos(a5) - (x - a6) * sin(a5))^2
//                       ) )
// =>
// a1 = C
// a2 = A
// a3 = (4 ln(2) / x_w^2)
// a4 = (4 ln(2) / y_w^2)
// a5 = b
// a6 = x_0
// a7 = y_0
// <=
// C = a1
// A = a2
// x_w = 2 sqrt( ln(2) / a3 )
// y_w = 2 sqrt( ln(2) / a4 )
// b = a5
// x_0 = a6
// y_0 = a7

// errors from covariance matrix
// y(x_1, ..., x_i) => e_y^2 = sum_i( (d/dx_i y(...))^2 * e_x_i^2)
// e_C^2 = e_a1^2
// e_A^2 = e_a2^2
// e_x_w^2 = ( ln(2) / |a3^(3)| ) * e_a3^2
// e_y_w^2 = ( ln(2) / |a4^(3)| ) * e_a4^2
// e_b^2 = e_a5^2
// e_x_0^2 = e_a6^2
// e_y_0^2 = e_a7^2

// Hesse
// df / da1 = 1
// df / da2 = exp(...)
// df / da3 = a2 * exp(...) * (-1) *
//            ((x - a6) * cos(a5) + (y - a7) * sin(a5))^2
// df / da4 = a2 * exp(...) * (-1) *
//            ((y - a7) * cos(a5) - (x - a6) * sin(a5))^2
// df / da5 = a2 * exp(...) * (-1) * 2 *
//            ( a3 * ((x - a6) * cos(a5) + (y - a7) * sin(a5)) *
//              ((y - a7) * cos(a5) - (x - a6) * sin(a5)) +
//              a4 * ((y - a7) * cos(a5) - (x - a6) * sin(a5)) *
//              (-1) * ((x - a6) * cos(a5) + (y - a7) * sin(a5))
//              )
// df / da6 = a2 * exp(...) * 2 *
//            ( a3 * ((x - a6) * cos(a5) + (y - a7) * sin(a5)) * cos(a5) -
//              a4 * ((y - a7) * cos(a5) - (x - a6) * sin(a5)) * sin(a5) )
// df / da7 = a2 * exp(...) * 2 *
//            ( a3 * ((x - a6) * cos(a5) + (y - a7) * sin(a5)) * sin(a5) +
//              a4 * ((y - a7) * cos(a5) - (x - a6) * sin(a5)) * cos(a5) )

template<class TPAR, class TDAT>
class Gaussian<TPAR, TDAT, 7, 2> : public MRQFunction<TPAR, TDAT, 7, 2>
{
   public:

      inline Gaussian() { }

      inline static FVector<TPAR, 7>
      partofit(const FVector<TPAR, 7>& restrict_ parameter)
      {
         FVector<TPAR, 7> fitpar;
         fitpar(1) = parameter(1); // start constant surface
         fitpar(2) = parameter(2); // start amplitude
         const TPAR M_4LN2 = TPAR(4.0 * M_LN2);
         const TPAR par3(parameter(3)); // xwidth, hyperbel also
         fitpar(3) = copysign(M_4LN2 / pow2(par3), par3);
         const TPAR par4(parameter(4)); // ywidth, hyperbel also
         fitpar(4) = copysign(M_4LN2 / pow2(par4), par4);
         fitpar(5) = parameter(5) * M_PI / 180.0; // start angle
         fitpar(6) = parameter(6); // start x-position
         fitpar(7) = parameter(7); // start y-position
         return fitpar;
      }
      
      inline static FVector<TPAR, 7>
      fittopar(const FVector<TPAR, 7>& fitpar,
               const typename FMatrix<TPAR, 7, 7>::TraceVector& trace)
      {
         FVector<TPAR, 7> parameter;
         parameter(1) = fitpar(1);
         parameter(2) = fitpar(2);
         const TPAR convwidth = TPAR(2.0 * ::sqrt(M_LN2));
         const TPAR fp3(fitpar(3)); // xwidth, hyp. also
         parameter(3) = copysign(convwidth / ::sqrt(::fabs(fp3)), fp3);
         const TPAR fp4(fitpar(4)); // xwidth, hyp. also
         parameter(4) = copysign(convwidth / ::sqrt(::fabs(fp4)), fp4);
         const TPAR fp5(fitpar(5));
         parameter(5) = ( fp5 -
                          (M_PI * (::ceil( fp5 / M_PI ) - 1.0)) ) *
            180.0 / M_PI;
         parameter(6) = fitpar(6);
         parameter(7) = fitpar(7);
         return parameter;
      }
      
      inline static FVector<TPAR, 7>
      covtoerr(const typename FMatrix<TPAR, 7, 7>::TraceVector& trace,
               const FVector<TPAR, 7>& fitpar)
      {
         FVector<TPAR, 7> error;
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         error(1) = trace(1);
         error(2) = trace(2);
         error(3) = trace(3) * M_LN2 / ::fabs(pow3(fitpar(3))); // hyperbel also
         error(4) = trace(4) * M_LN2 / ::fabs(pow3(fitpar(4))); // hyperbel also
         error(5) = trace(5) * 180.0 / M_PI;
         error(6) = trace(6);
         error(7) = trace(7);
         return error;
      }
      
      inline static void fill(const FVector<TPAR, 7>& restrict_ invalue,
                              MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 7> fitpar(partofit(invalue));
         const TPAR bx( fitpar(3) ); // xwidth
         const TPAR by( fitpar(4) ); // ywidth
         const TPAR cw( ::cos(fitpar(5)) ); // cos of angle
         const TPAR sw( ::sin(fitpar(5)) ); // sin of angle
         a = fitpar(1) + fitpar(2) *
            exp( -( (bx * pow2( (indexPosDbl(a, 1) - fitpar(6)) * cw +
                                (indexPosDbl(a, 2) - fitpar(7)) * sw )) +
                    (by * pow2( (indexPosDbl(a, 2) - fitpar(7)) * cw -
                                (indexPosDbl(a, 1) - fitpar(6)) * sw )) ) );
      }
      
      TPAR marquardtCoefficients(const FVector<TPAR, 7>& restrict_ parameter,
                                 const TPAR chisquare_limit,
                                 FMatrix<TPAR, 7, 7>& restrict_ a,
                                 FVector<TPAR, 7>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( M_LN2 * TPAR(DBL_MAX_EXP - 1));
         const TPAR le_limit( M_LN2 * TPAR(DBL_MIN_EXP - 1));

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR bx(parameter(3)); // xwidth
         const TPAR by(parameter(4)); // ywidth
         const TPAR bymbx(by - bx);
         const TPAR cw(::cos(parameter(5))); // cos of angle
         const TPAR sw(::sin(parameter(5))); // sin of angle
         const TPAR x0(parameter(6)); // xposition
         const TPAR y0(parameter(7)); // yposition

         tMatClearHesse<TPAR, 7>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 7> derivative;
         derivative(1) = TPAR(1);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();

         TPAR chisquare = TPAR(0);

         for(int yy = this->data_.minIndex(2); yy <= this->data_.maxIndex(2); ++yy)
         {
            const TPAR ydif = TPAR(yy) - y0;
            const TPAR ysw = ydif * sw;
            const TPAR ycw = ydif * cw;
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               const TDAT dat = *data_i;
               const TDAT er2 = *error2_i;
               if( (dat != this->nan_data_) && (er2 > TDAT(0)) )
               {
                  const TPAR xdif = TPAR(xx) - x0;
                  const TPAR t1 = xdif * cw + ysw;
                  const TPAR t2 = ycw - xdif * sw;
                  const TPAR t1s = t1 * t1;
                  const TPAR t2s = t2 * t2;
                  TPAR ert = -(bx * t1s + by * t2s);
                  if(ert > ue_limit) ert = ue_limit;
                  else if(ert < le_limit) ert = le_limit;
                  ert = ::exp(ert);
                  derivative(2) = ert;
                  const TPAR yg = am * ert;
                  derivative(3) = -(t1s * yg);
                  derivative(4) = -(t2s * yg);
                  const TPAR yg2 = TPAR(2.0) * yg;
                  derivative(5) = bymbx * t1 * t2 * yg2;
                  const TPAR bxt1 = bx * t1;
                  const TPAR byt2 = by * t2;
                  derivative(6) = (bxt1 * cw - byt2 * sw) * yg2;
                  derivative(7) = (bxt1 * sw + byt2 * cw) * yg2;
                  const TPAR df = TPAR(dat - TDAT(yg + sf));
                  const TPAR sig2 = TPAR(er2);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 7>::eval(derivative, df, sig2, a, b);
               }
               ++data_i;
               ++error2_i;
            }
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 7>::eval(a);
         return chisquare;
      }

};

template<class TPAR, class TDAT>
class Gaussian<TPAR, TDAT, 5, 2> : public MRQFunction<TPAR, TDAT, 5, 2>
{
   public:

      inline Gaussian() { }

      inline void setData(const MArray<TDAT, 2>& indata,
                          const TDAT in_nan,
                          const MArray<TDAT, 2>& inerror2)
      {
         if((indata.nelements()%2) != 1)
            throw LinearAlgebraException("This fit works only on odd sized MArrays.");
         MRQFunction<TPAR, TDAT, 5, 2>::setData(indata, in_nan, inerror2);
         this->data_.setBase( -(indata.length(1)/2), -(indata.length(2)/2) );
         rdata_.makeReference(indata);
         rdata_.reverseSelf(1); rdata_.reverseSelf(2);
         rerror2_.makeReference(inerror2);
         rerror2_.reverseSelf(1); rerror2_.reverseSelf(2);
      }

      inline void freeData()
      {
         MRQFunction<TPAR, TDAT, 5, 2>::freeData();
         rdata_.free();
         rerror2_.free();
      }

      inline static FVector<TPAR, 5>
      partofit(const FVector<TPAR, 5>& parameter)
      {
         FVector<TPAR, 5> fitpar;
         fitpar(1) = parameter(1); // start constant surface
         fitpar(2) = parameter(2); // start amplitude
         //const TPAR convwidth = TPAR(2.0 * sqrt(M_LN2)); // ellipse only
         //fitpar(3) = convwidth / fabs(parameter(3)); // xwidth, ellipse only
         //fitpar(4) = convwidth / fabs(parameter(4)); // ywidth, ellipse only
         const TPAR M_4LN2 = TPAR(4.0 * M_LN2);
         const TPAR par3(parameter(3)); // xwidth, hyperbel also
         fitpar(3) = copysign(M_4LN2 / pow2(par3), par3);
         const TPAR par4(parameter(4)); // ywidth, hyperbel also
         fitpar(4) = copysign(M_4LN2 / pow2(par4), par4);
         fitpar(5) = parameter(5) * (M_PI / 180.0); // start angle
         return fitpar;
      }
      
      inline static FVector<TPAR, 5>
      fittopar(const FVector<TPAR, 5>& fitpar,
               const typename FMatrix<TPAR, 5, 5>::TraceVector& trace)
      {
         FVector<TPAR, 5> parameter;
         parameter(1) = fitpar(1);
         parameter(2) = fitpar(2);
         const TPAR convwidth = TPAR(2.0 * ::sqrt(M_LN2));
         //parameter(3) = convwidth / fabs(fitpar(3)); // xwidth, ellipse only
         //parameter(4) = convwidth / fabs(fitpar(4)); // ywidth, ellipse only
         const TPAR fp3(fitpar(3)); // xwidth, hyp. also
         parameter(3) = copysign(convwidth / ::sqrt(::fabs(fp3)), fp3);
         const TPAR fp4(fitpar(4)); // xwidth, hyp. also
         parameter(4) = copysign(convwidth / ::sqrt(::fabs(fp4)), fp4);

//          if( fitpar(3) >= TPAR(0.0) )
//             parameter(3) = convwidth / sqrt(fitpar(3));  // xwidth, ellipse and
//          else
//             parameter(3) = -convwidth / sqrt(-fitpar(3));  // xwidth, hyperbel also
//          if( fitpar(4) >= TPAR(0.0) )
//             parameter(4) = convwidth / sqrt(fitpar(4));  // xwidth, ellipse and
//          else
//             parameter(4) = -convwidth / sqrt(-fitpar(4));  // xwidth, hyperbel also
         const TPAR fp5(fitpar(5));
         parameter(5) = ( fp5 -
                          (M_PI * (::ceil( fp5 / M_PI ) - 1.0)) ) *
            (180.0 / M_PI);
         return parameter;
      }
      
      inline static FVector<TPAR, 5>
      covtoerr(const typename FMatrix<TPAR, 5, 5>::TraceVector& trace,
               const FVector<TPAR, 5>& fitpar)
      {
         FVector<TPAR, 5> error;
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         error(1) = trace(1);
         error(2) = trace(2);
         //error(3) = trace(3) * (4.0 * M_LN2) / pow4(fitpar(3)); // ellipse only
         //error(4) = trace(4) * (4.0 * M_LN2) / pow4(fitpar(4)); // ellipse only
         error(3) = trace(3) * M_LN2 / ::fabs(pow3(fitpar(3))); // hyperbel also
         error(4) = trace(4) * M_LN2 / ::fabs(pow3(fitpar(4))); // hyperbel also
         error(5) = trace(5) * 180.0 / M_PI;
         return error;
      }
      
      inline static MArray<TDAT, 2>&
      fill(const FVector<TPAR, 5>& restrict_ invalue,
	   const int x, const int y,
	   MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 5> fitpar(partofit(invalue));
         const TPAR bx(fitpar(3)); // xwidth
         const TPAR by(fitpar(4)); // ywidth
         const TPAR cw( ::cos(fitpar(5)) ); // cos of angle
         const TPAR sw( ::sin(fitpar(5)) ); // sin of angle
         a = fitpar(1) + fitpar(2) *
            exp( -( (bx * pow2( (indexPosDbl(a, 1) - x) * cw +
                                (indexPosDbl(a, 2) - y) * sw )) +
                    (by * pow2( (indexPosDbl(a, 2) - y) * cw -
                                (indexPosDbl(a, 1) - x) * sw )) ) );
	 return a;
      }

      inline static MArray<TDAT, 2>&
      fillExp(const FVector<TPAR, 5>& restrict_ invalue,
	      const int x, const int y,
	      MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 5> fitpar(partofit(invalue));
         const TPAR bx(fitpar(3)); // xwidth
         const TPAR by(fitpar(4)); // ywidth
         const TPAR cw( ::cos(fitpar(5)) ); // cos of angle
         const TPAR sw( ::sin(fitpar(5)) ); // sin of angle
         a = TDAT( -( (bx * pow2( TPAR(indexPosInt(a, 1) - x) * cw +
                                  TPAR(indexPosInt(a, 2) - y) * sw )) +
                      (by * pow2( TPAR(indexPosInt(a, 2) - y) * cw -
                                  TPAR(indexPosInt(a, 1) - x) * sw )) ) );
	 return a;
      }

      inline TPAR marquardtCoefficients(const FVector<TPAR, 5>& restrict_ parameter,
                                        const TPAR chisquare_limit,
                                        FMatrix<TPAR, 5, 5>& restrict_ a,
                                        FVector<TPAR, 5>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( M_LN2 * TPAR(DBL_MAX_EXP - 1));
         const TPAR le_limit( M_LN2 * TPAR(DBL_MIN_EXP - 1));
         //const TPAR ue_limit( 18.0 );
         //const TPAR le_limit( -14.0 );

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR bx(parameter(3)); // xwidth
         const TPAR by(parameter(4)); // ywidth
         //const TPAR bx2(bx*bx); // ellipse only
         //const TPAR by2(by*by); // ellipse only
         //const TPAR by2mbx2(by2 - bx2); // ellipse only
         //const TPAR b2x(TPAR(2.0) * bx); // ellipse only
         //const TPAR b2y(TPAR(2.0) * by); // ellipse only
         const TPAR bymbx = by - bx; // hyperbel also
         const TPAR cw(::cos(parameter(5))); // cos of angle
         const TPAR sw(::sin(parameter(5))); // sin of angle

         tMatClearHesse<TPAR, 5>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 5> derivative;
         derivative(1) = TPAR(1.0);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         //typename MArray<TDAT, 2>::IndexIterator index_i = data_.indexBegin();
         typename MArray<TDAT, 2>::const_iterator rdata_i = rdata_.begin();
         
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();
         typename MArray<TDAT, 2>::const_iterator rerror2_i = rerror2_.begin();

         TPAR chisquare = TPAR(0);

         for(int yy = this->data_.minIndex(2); yy < 0; ++yy)
         {
            const TPAR ydif = TPAR(yy);
            const TPAR ysw = ydif * sw;
            const TPAR ycw = ydif * cw;
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               const bool have_data = (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
               const bool have_rdata = (*rdata_i != this->nan_data_) && (*rerror2_i > TDAT(0));
               if( have_data || have_rdata )
               {
                  const TPAR xdif = TPAR(xx);
                  const TPAR t1 = xdif * cw + ysw;
                  const TPAR t2 = ycw - xdif * sw;
                  const TPAR t1s = t1 * t1;
                  const TPAR t2s = t2 * t2;
                  //TPAR ert = -(bx2 * t1s + by2 * t2s); // ellipse only
                  TPAR ert = -(bx * t1s + by * t2s); // hyperbel also
                  if(ert > ue_limit) ert = ue_limit;
                  else if(ert < le_limit) ert = le_limit;
                  const TPAR bert = ::exp(ert);
                  derivative(2) = bert;
                  const TPAR yg = am * bert;
                  //derivative(3) = -(b2x * t1s * yg); // ellipse only
                  //derivative(4) = -(b2y * t2s * yg); // ellipse only
                  //derivative(5) = (by2mbx2) * t1 * t2 * yg * TPAR(2.0); // ellipse only
                  derivative(3) = -(t1s * yg); // hyperbel only
                  derivative(4) = -(t2s * yg); // hyperbel only
                  derivative(5) = (bymbx) * t1 * t2 * yg * TPAR(2.0); // hyperbel only
                  if(have_data)
                  {
                     const TPAR df = TPAR(*data_i - TDAT(yg + sf));
                     const TPAR sig2 = TPAR(*error2_i);
                     chisquare += df * df / sig2;
                     if( (chisquare > chisquare_limit) &&
                         (chisquare_limit != TPAR(0)) )
                        return chisquare;
                     tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
                  }
                  if(have_rdata)
                  {
                     const TPAR df = TPAR(*rdata_i - TDAT(yg + sf));
                     const TPAR sig2 = TPAR(*rerror2_i);
                     chisquare += df * df / sig2;
                     if( (chisquare > chisquare_limit) &&
                         (chisquare_limit != TPAR(0)) )
                        return chisquare;
                     tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
                  }                  

               }
               ++data_i; ++rdata_i;
               ++error2_i; ++rerror2_i;
            }
         }
         for(int xx = this->data_.minIndex(1); xx < 0; ++xx)
         {
            const bool have_data = (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
            const bool have_rdata = (*rdata_i != this->nan_data_) && (*rerror2_i > TDAT(0));
            if( have_data || have_rdata )
            {
               const TPAR xdif = TPAR(xx);
               const TPAR t1 = xdif * cw;
               const TPAR t2 = - xdif * sw;
               const TPAR t1s = t1 * t1;
               const TPAR t2s = t2 * t2;
               //TPAR ert = -(bx2 * t1s + by2 * t2s); // ellipse only
               TPAR ert = -(bx * t1s + by * t2s); // hyperbel also
               if(ert > ue_limit) ert = ue_limit;
               else if(ert < le_limit) ert = le_limit;
               const TPAR bert = ::exp(ert);
               derivative(2) = bert;
               const TPAR yg = am * bert;
               //derivative(3) = -(b2x * t1s * yg); // ellipse only
               //derivative(4) = -(b2y * t2s * yg); // ellipse only
               //derivative(5) = (by2mbx2) * t1 * t2 * yg * TPAR(2.0); // ellipse only
               derivative(3) = -(t1s * yg); // hyperbel only
               derivative(4) = -(t2s * yg); // hyperbel only
               derivative(5) = (bymbx) * t1 * t2 * yg * TPAR(2.0); // hyperbel only
               if(have_data)
               {
                  const TPAR df = TPAR(*data_i - TDAT(yg + sf));
                  const TPAR sig2 = TPAR(*error2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
               }
               if(have_rdata)
               {
                  const TPAR df = TPAR(*rdata_i - TDAT(yg + sf));
                  const TPAR sig2 = TPAR(*rerror2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
               }                  
            }
            ++data_i; ++rdata_i;
            ++error2_i; ++rerror2_i;
         }
         if( (*data_i != this->nan_data_) && (*error2_i > TDAT(0)) )
         {
            //derivative(2) = TPAR(1.0);
            //derivative(3) = TPAR(0.0);
            //derivative(4) = TPAR(0.0);
            //derivative(5) = TPAR(0.0);

            const TPAR df = TPAR(*data_i - TDAT(am + sf));
            const TPAR sig2 = TPAR(*error2_i);
            chisquare += df * df / sig2;
            if( (chisquare > chisquare_limit) &&
                (chisquare_limit != TPAR(0)) )
               return chisquare;
            tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
            // means:
            //for(int l=1; l <= 5; ++l)
            //{
            //const TPAR wt = derivative(l) / sig2;
            //for(int k=1; k <= l; ++k)
            //a(l, k) += wt * derivative(k);
            //b(l) += df * wt;
            //}
            // so only next are left
            const TPAR sig2i = TPAR(1.0) / sig2;
            a(1,1) += sig2i;
            a(2,1) += sig2i;
            a(2,2) += sig2i;
            const TPAR dfsig2i = df * sig2i;
            b(1) += dfsig2i;
            b(2) += dfsig2i;
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 5>::eval(a);
         return chisquare;
      }

   private:
      MArray<TDAT, 2> rdata_;
      MArray<TDAT, 2> rerror2_;
};

template<class TPAR, class TDAT>
class Gaussian<TPAR, TDAT, 3, 2> : public MRQFunction<TPAR, TDAT, 3, 2>
{
   public:

      inline Gaussian() { }

      inline void setData(const MArray<TDAT, 2>& indata,
                          const TDAT in_nan,
                          const MArray<TDAT, 2>& inerror2)
      {
         if((indata.nelements()%2) != 1)
            throw LinearAlgebraException("This fit works only on odd sized MArrays.");
         MRQFunction<TPAR, TDAT, 3, 2>::setData(indata, in_nan, inerror2);
         this->data_.setBase( -(indata.length(1)/2), -(indata.length(2)/2) );
         rdata_.makeReference(indata);
         rdata_.reverseSelf(1); rdata_.reverseSelf(2);
         rerror2_.makeReference(inerror2);
         rerror2_.reverseSelf(1); rerror2_.reverseSelf(2);
      }

      inline void freeData()
      {
         MRQFunction<TPAR, TDAT, 3, 2>::freeData();
         rdata_.free();
         rerror2_.free();
      }

      inline static FVector<TPAR, 3>
      partofit(const FVector<TPAR, 3>& parameter)
      {
         FVector<TPAR, 3> fitpar;
         fitpar(1) = parameter(1); // start constant surface
         fitpar(2) = parameter(2); // start amplitude
         //const TPAR convwidth = TPAR(2.0 * sqrt(M_LN2)); // circle only
         //fitpar(3) = convwidth / fabs(parameter(3)); // width, circle only
         const TPAR M_4LN2 = TPAR(4.0 * M_LN2);
         const TPAR par3(parameter(3)); // xwidth, hyperbel also
         fitpar(3) = copysign(M_4LN2 / pow2(par3), par3);
         return fitpar;
      }
      
      inline static FVector<TPAR, 3>
      fittopar(const FVector<TPAR, 3>& fitpar,
               const typename FMatrix<TPAR, 3, 3>::TraceVector& trace)
      {
         FVector<TPAR, 3> parameter;
         parameter(1) = fitpar(1);
         parameter(2) = fitpar(2);
         const TPAR convwidth = TPAR(2.0 * ::sqrt(M_LN2));
         //parameter(3) = convwidth / fabs(fitpar(3)); // width, circle only
         const TPAR fp3(fitpar(3)); // xwidth, hyp. also
         parameter(3) = copysign(convwidth / ::sqrt(::fabs(fp3)), fp3);
         return parameter;
      }
      
      inline static FVector<TPAR, 3>
      covtoerr(const typename FMatrix<TPAR, 3, 3>::TraceVector& trace,
               const FVector<TPAR, 3>& fitpar)
      {
         FVector<TPAR, 3> error;
         // error(i) = sqrt( sum_j(pow2(d_par(j) / d_fitpar(j)) * (trace(j))) );
         error(1) = trace(1);
         error(2) = trace(2);
         // error(3) = ( 4.0 * M_LN2 / pow4(fitpar(3)) ) * trace(3) ;
         //error(3) = trace(3) * (4.0 * M_LN2) / pow4(fitpar(3)); // ellipse only
         error(3) = trace(3) * M_LN2 / ::fabs(pow3(fitpar(3))); // hyperbel also
         return error;
      }
      
      inline TPAR marquardtCoefficients(const FVector<TPAR, 3>& restrict_ parameter,
                                        const TPAR chisquare_limit,
                                        FMatrix<TPAR, 3, 3>& restrict_ a,
                                        FVector<TPAR, 3>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( M_LN2 * TPAR(DBL_MAX_EXP - 1));
         const TPAR le_limit( M_LN2 * TPAR(DBL_MIN_EXP - 1));

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR br(parameter(3)); // width
         //const TPAR br2(br*br); // ellipse only
         //const TPAR b2r(TPAR(2.0) * br); // ellipse only

         tMatClearHesse<TPAR, 3>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 3> derivative;
         derivative(1) = TPAR(1.0);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         //typename MArray<TDAT, 2>::IndexIterator index_i = data_.indexBegin();
         typename MArray<TDAT, 2>::const_iterator rdata_i = rdata_.begin();
         
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();
         typename MArray<TDAT, 2>::const_iterator rerror2_i = rerror2_.begin();

         TPAR chisquare = TPAR(0);

         for(int yy = this->data_.minIndex(2); yy < 0; ++yy)
         {
            const TPAR yd2 = TPAR(yy * yy);
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               const bool have_data = (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
               const bool have_rdata = (*rdata_i != this->nan_data_) && (*rerror2_i > TDAT(0));
               if( have_data || have_rdata )
               {
                  const TPAR mr2 = -(yd2 + TPAR(xx * xx));
                  //TPAR ert = br2 * mr2; // ellipse only
                  TPAR ert = br * mr2; // hyperbel also
                  if(ert > ue_limit) ert = ue_limit;
                  else if(ert < le_limit) ert = le_limit;
                  const TPAR bert = ::exp(ert);
                  derivative(2) = bert;
                  const TPAR yg = am * bert;
                  //derivative(3) = b2r * mr2 * yg; // ellipse only
                  derivative(3) = yg * mr2; // hyperbel only
                  if(have_data)
                  {
                     const TPAR df = TPAR(*data_i - TDAT(yg + sf));
                     const TPAR sig2 = TPAR(*error2_i);
                     chisquare += df * df / sig2;
                     if( (chisquare > chisquare_limit) &&
                         (chisquare_limit != TPAR(0)) )
                        return chisquare;
                     tMatHesse<TPAR, 3>::eval(derivative, df, sig2, a, b);
                  }
                  if(have_rdata)
                  {
                     const TPAR df = TPAR(*rdata_i - TDAT(yg + sf));
                     const TPAR sig2 = TPAR(*rerror2_i);
                     chisquare += df * df / sig2;
                     if( (chisquare > chisquare_limit) &&
                         (chisquare_limit != TPAR(0)) )
                        return chisquare;
                     tMatHesse<TPAR, 3>::eval(derivative, df, sig2, a, b);
                  }                  

               }
               ++data_i; ++rdata_i;
               ++error2_i; ++rerror2_i;
            }
         }
         for(int xx = this->data_.minIndex(1); xx < 0; ++xx)
         {
            const bool have_data = (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
            const bool have_rdata = (*rdata_i != this->nan_data_) && (*rerror2_i > TDAT(0));
            if( have_data || have_rdata )
            {
               const TPAR mr2 = -(TPAR(xx * xx));
               //TPAR ert = br2 * mr2; // ellipse only
               TPAR ert = br * mr2; // hyperbel also
               if(ert > ue_limit) ert = ue_limit;
               else if(ert < le_limit) ert = le_limit;
               const TPAR bert = ::exp(ert);
               derivative(2) = bert;
               const TPAR yg = am * bert;
               //derivative(3) = b2r * mr2 * yg; // ellipse only
               derivative(3) = yg * mr2; // hyperbel only
               if(have_data)
               {
                  const TPAR df = TPAR(*data_i - TDAT(yg + sf));
                  const TPAR sig2 = TPAR(*error2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 3>::eval(derivative, df, sig2, a, b);
               }
               if(have_rdata)
               {
                  const TPAR df = TPAR(*rdata_i - TDAT(yg + sf));
                  const TPAR sig2 = TPAR(*rerror2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 3>::eval(derivative, df, sig2, a, b);
               }                  
            }
            ++data_i; ++rdata_i;
            ++error2_i; ++rerror2_i;
         }
         if( (*data_i != this->nan_data_) && (*error2_i > TDAT(0)) )
         {
            //derivative(2) = TPAR(1.0);
            //derivative(3) = TPAR(0.0);

            const TPAR df = TPAR(*data_i - TDAT(am + sf));
            const TPAR sig2 = TPAR(*error2_i);
            chisquare += df * df / sig2;
            if( (chisquare > chisquare_limit) &&
                (chisquare_limit != TPAR(0)) )
               return chisquare;
            //tMatHesse<TPAR, 3>::eval(derivative, df, sig2, a, b);
            // means:
            //for(int l=1; l <= 3; ++l)
            //{
            //const TPAR wt = derivative(l) / sig2;
            //for(int k=1; k <= l; ++k)
            //a(l, k) += wt * derivative(k);
            //b(l) += df * wt;
            //}
            // so only next are left
            const TPAR sig2i = TPAR(1.0) / sig2;
            a(1,1) += sig2i;
            a(2,1) += sig2i;
            a(2,2) += sig2i;
            const TPAR dfsig2i = df * sig2i;
            b(1) += dfsig2i;
            b(2) += dfsig2i;
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 3>::eval(a);
         return chisquare;
      }

   private:
      MArray<TDAT, 2> rdata_;
      MArray<TDAT, 2> rerror2_;
};



// polynomial version

template<class TPAR, class TDAT, int NPAR, int NDIM>
class PolyGaussian : public MRQFunction<TPAR, TDAT, NPAR, NDIM> { };

// 7 parameter Gaussian
// f(x, y) = C + A * exp( -( (4 ln(2) / x_w^2) *
//                           ( (x - x_0) * cos(b) + (y - y_0) * sin(b) )^2 +
//                           (4 ln(2) / y_w^2) *
//                           ( (y - y_0) * cos(b) - (x - x_0) * sin(b) )^2 ) )
// -> Polynomial version
// f(x, y) = a1 + exp( -(a2 + a3 * (x - a6)^2 +
//                            a4 * (y - a7)^2 +
//                            a5 * (x - a6) * (y - a7) ) )
// =>
// a1 = C
// a2 = - ln A
// a3 = (4 ln(2) / x_w^2) * cos^2(b) + (4 ln(2) / y_w^2) * sin^2(b)
// a4 = (4 ln(2) / y_w^2) * cos^2(b) + (4 ln(2) / x_w^2) * sin^2(b)
// a5 = 2 * sin(b) * cos(b) * ((4 ln(2) / x_w^2) - (4 ln(2) / y_w^2))
// a6 = x_0
// a7 = y_0
// <= still has some ambiguities
// C = a1
// A = exp(-a2)
// x_w = 2 sqrt( ln(2) / ( a3 + a4 +
//                         (a3 - a4) / cos( arctan( a5 / (a3 - a4) ) ) ) )
// y_w = 2 sqrt( ln(2) / ( a3 + a4 -
//                         (a3 - a4) / cos( arctan( a5 / (a3 - a4) ) ) ) )
// b = arctan(a5 / (a3 - a4)) / 2 
// x_0 = a6
// y_0 = a7

// errors from covariance matrix
// y(x_1, ..., x_i) => e_y^2 = sum_i( (d/dx_i y(...))^2 * e_x_i^2)
// e_C^2 = e_a1^2
// e_A^2 = (-exp(-a2))^2 * e_a2^2 = A^2 * e_a2^2
// e_x_w^2 = 
// e_y_w^2 = 
// e_b^2 = [ -a5^2 * e_a3^2 + a5^2 * e_a4^2 + (a3 - a4)^2 * e_a5^2 ] *
//         [ 0.5 / ((a3 - a4)^2 + a5^2) ]^2

// df / da1 = 1
// df / da2 = exp(...) * (-1)
// df / da3 = exp(...) * (-1) * (x - a6)^2
// df / da4 = exp(...) * (-1) * (y - a7)^2
// df / da5 = exp(...) * (-1) * (x - a6) * (y - a7)
// df / da6 = exp(...) * (-1) * (2 * a3 * (-1) (x - a6) + (-1) * a5 (y - a7)
// df / da7 = exp(...) * (-1) * (2 * a4 * (-1) (y - a7) + (-1) * a5 (x - a6)

template<class TPAR, class TDAT>
class PolyGaussian<TPAR, TDAT, 7, 2> : public MRQFunction<TPAR, TDAT, 7, 2>
{
   public:

      inline PolyGaussian() { }

      inline static FVector<TPAR, 7>
      partofit(const FVector<TPAR, 7>& parameter)
      {
         FVector<TPAR, 7> fitpar;
         fitpar(1) = parameter(1); // start constant surface
	 if(parameter(2)<=0)
            throw LinearAlgebraException("invalid start amplitude (<=0) for PolyGaussian fit in mrqfuncs.h.");
         fitpar(2) = -::log(parameter(2)); // start amplitude
         const TPAR M_4LN2 = 4.0 * M_LN2;
         const TPAR par3(parameter(3));
         const TPAR sx = 
            copysign(M_4LN2 / pow2(par3), par3); // xwidth
         const TPAR par4(parameter(4));
         const TPAR sy = 
            copysign(M_4LN2 / pow2(par4), par4); // ywidth
         const TPAR w = parameter(5) * (M_PI / 180.0); // start angle
         const TPAR cw = ::cos(w);
         const TPAR sw = ::sin(w);
         fitpar(5) = 2.0 * cw * sw * (sx - sy);
         const TPAR cw2 = pow2(cw);
         const TPAR sw2 = pow2(sw);         
         fitpar(3) = sx * cw2 + sy * sw2; // xwidth
         fitpar(4) = sx * sw2 + sy * cw2; // ywidth
         fitpar(6) = parameter(6); // start x-position
         fitpar(7) = parameter(7); // start y-position
         return fitpar;
      }

      inline static FVector<TPAR, 7>
      fittopar(const FVector<TPAR, 7>& fitpar,
               const typename FMatrix<TPAR, 7, 7>::TraceVector& trace)
      {
         FVector<TPAR, 7> parameter;
         parameter(1) = fitpar(1);
         parameter(2) = ::exp(-fitpar(2));
         const TPAR qdif = fitpar(3) - fitpar(4);
         const TPAR qsum = fitpar(3) + fitpar(4);

         TPAR w2, qdcw;
         const TPAR error_limit2 = TPAR(4.0);
         const TPAR qdif2 = pow2(qdif);
         const TPAR qdiferr2 = trace(3) + trace(4);
         const TPAR fp52  = pow2( fitpar(5) );
         const TPAR fp5err2  = trace(5);

         if( qdif2 < (error_limit2 * qdiferr2) ) // rather round in x to y
         {
            if( (fp52 < (error_limit2 * fp5err2)) &&
                (fp52 < qdif2) ) // and not diagonal
            {
               w2 = TPAR(0.0);
               qdcw = qdif;
            }
            else // but still diagonal
            {
               //w2 = ::atan2(fitpar(5), copysign(0.0, qdif));
               //qdcw = -copysign(fitpar(5), qdif);
               w2 = ::atan2( fitpar(5), qdif );
               qdcw = qdif / ::cos( w2 );
            }
         }
         else
         {
            w2 = ::atan2( fitpar(5), qdif );
            qdcw = qdif / ::cos( w2 );
         }
         //parameter(5) = w2 / TPAR(2.0 * M_PI / 180.0);
         parameter(5) = w2 * TPAR(90.0 / M_PI);

         const TPAR sx = (qsum + qdcw) / TPAR(2.0);
         const TPAR sy = (qsum - qdcw) / TPAR(2.0);
         parameter(3) = (sx >= TPAR(0.0)) ? 
            TPAR(2.0 * ::sqrt(M_LN2 / sx)) :
            -TPAR(2.0 * ::sqrt(-M_LN2 / sx));
         parameter(4) = (sy >= TPAR(0.0)) ? 
            TPAR(2.0 * ::sqrt(M_LN2 / sy)) :
            -TPAR(2.0 * ::sqrt(-M_LN2 / sy));

         parameter(6) = fitpar(6);
         parameter(7) = fitpar(7);
         return parameter;
      }
      
      inline FVector<TPAR, 7>
      covtoerr(const typename FMatrix<TPAR, 7, 7>::TraceVector& trace,
               const FVector<TPAR, 7>& fitpar)
      {
         //throw LinearAlgebraException("error conversion still missing ...");
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         
         // !!! error propagation is too complicated -> use a trick
         Gaussian<TPAR, TDAT, 7, 2> simplegauss;
         simplegauss.setData(this->data_, this->nan_data_, this->error2_);
         FVector<TPAR, 7> parameter = simplegauss.partofit(fittopar(fitpar, trace)); 
         FMatrix<TPAR, 7, 7> cov;
         FVector<TPAR, 7> dummy;
         simplegauss.marquardtCoefficients(parameter,
                                           TPAR(0.0), cov, dummy);
         GaussJ<TPAR, 7>::eval(cov, dummy);

         //error(1) = trace(1);
         //const TPAR ea22 = pow2( exp(-fitpar(2)) );
         //error(2) = ea22 * trace(2);
         // ...
         //error(3) = trace(3);
         //error(4) = trace(4);
         //error(5) = trace(5);
         //error(6) = trace(6);
         //error(7) = trace(7);
         return simplegauss.covtoerr(cov.traceVector(), parameter);
      }
      
      inline TPAR marquardtCoefficients(const FVector<TPAR, 7>& restrict_ parameter,
                                        const TPAR chisquare_limit,
                                        FMatrix<TPAR, 7, 7>& restrict_ a,
                                        FVector<TPAR, 7>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( M_LN2 * TPAR(DBL_MAX_EXP - 1));
         const TPAR le_limit( M_LN2 * TPAR(DBL_MIN_EXP - 1));

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR bx(parameter(3)); // xwidth
         const TPAR by(parameter(4)); // ywidth
         const TPAR bxy(parameter(5)); // mixed width
         const TPAR x0(parameter(6)); // xposition
         const TPAR y0(parameter(7)); // yposition

         tMatClearHesse<TPAR, 7>::eval(a);
         b = TPAR(0.0);
         FVector<TPAR, 7> derivative;
         derivative(1) = TPAR(1.0);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();

         TPAR chisquare = TPAR(0.0);

         for(int yy = this->data_.minIndex(2); yy <= this->data_.maxIndex(2); ++yy)
         {
            const TPAR ydif  = TPAR(yy) - y0;
            const TPAR y2    = pow2(ydif);
            const TPAR y_bxy = ydif * bxy;
            const TPAR y_2by = TPAR(2.0) * ydif * by;
            const TPAR y2_by = y2 * by;
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               if( (*data_i != this->nan_data_) && (*error2_i > TDAT(0)) )
               {
                  const TPAR xdif = TPAR(xx) - x0;
                  const TPAR x2 = pow2(xdif);
                  const TPAR erta = -( x2 * bx + xdif * y_bxy + y2_by + am );
                  const TPAR eberta = (erta < le_limit) ?
                     TPAR(0.0) :
                     ( (erta > ue_limit) ?
                       ::exp(ue_limit) : ::exp(erta) );
                  const TPAR meberta = -eberta;
                  derivative(2) = meberta;
                  derivative(3) = meberta * x2;
                  derivative(4) = meberta * y2;
                  derivative(5) = meberta * xdif * ydif;
                  derivative(6) = eberta * ( TPAR(2.0) * xdif * bx + y_bxy );
                  derivative(7) = eberta * ( y_2by + xdif * bxy );
                  const TPAR df = TPAR(*data_i - TDAT(eberta + sf));
                  const TPAR sig2 = TPAR(*error2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0.0)) )
                     return chisquare;
                  tMatHesse<TPAR, 7>::eval(derivative, df, sig2, a, b);
               }
               ++data_i;
               ++error2_i;
            }
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 7>::eval(a);
         return chisquare;
      }

};

template<class TPAR, class TDAT>
class PolyGaussian<TPAR, TDAT, 5, 2> : public MRQFunction<TPAR, TDAT, 5, 2>
{
   public:

      inline PolyGaussian() { }

      inline void setData(const MArray<TDAT, 2>& indata,
                          const TDAT in_nan,
                          const MArray<TDAT, 2>& inerror2)
      {
         if((indata.nelements()%2) != 1)
            throw LinearAlgebraException("This fit works only on odd sized MArrays.");
         MRQFunction<TPAR, TDAT, 5, 2>::setData(indata, in_nan, inerror2);
         this->data_.setBase( -(indata.length(1)/2), -(indata.length(2)/2) );
         rdata_.makeReference(indata);
         rdata_.reverseSelf(1); rdata_.reverseSelf(2);
         rerror2_.makeReference(inerror2);
         rerror2_.reverseSelf(1); rerror2_.reverseSelf(2);
      }

      inline void freeData()
      {
         MRQFunction<TPAR, TDAT, 5, 2>::freeData();
         rdata_.free();
         rerror2_.free();
      }

      inline static FVector<TPAR, 5>
      partofit(const FVector<TPAR, 5>& parameter)
      {
         FVector<TPAR, 5> fitpar;
         fitpar(1) = parameter(1); // start constant surface
	 if(parameter(2)<=0)
            throw LinearAlgebraException("invalid start amplitude (<=0) for PolyGaussian fit in mrqfuncs.h.");
         fitpar(2) = -::log(parameter(2)); // start amplitude
         const TPAR M_4LN2 = 4.0 * M_LN2;
         const TPAR par3(parameter(3));
         const TPAR sx = 
            copysign(M_4LN2 / pow2(par3), par3); // xwidth
         const TPAR par4(parameter(4));
         const TPAR sy = 
            copysign(M_4LN2 / pow2(par4), par4); // ywidth
         const TPAR w = parameter(5) * (M_PI / 180.0); // start angle
         const TPAR cw = ::cos(w);
         const TPAR sw = ::sin(w);
         fitpar(5) = 2.0 * cw * sw * (sx - sy);
         const TPAR cw2 = pow2(cw);
         const TPAR sw2 = pow2(sw);         
         //fitpar(3) = sqrt(sx * cw2 + sy * sw2); // xwidth
         //fitpar(4) = sqrt(sx * sw2 + sy * cw2); // ywidth
         fitpar(3) = sx * cw2 + sy * sw2; // xwidth
         fitpar(4) = sx * sw2 + sy * cw2; // ywidth

         return fitpar;
      }
      
      inline static FVector<TPAR, 5>
      fittopar(const FVector<TPAR, 5>& fitpar,
               const typename FMatrix<TPAR, 5, 5>::TraceVector& trace)
      {
         FVector<TPAR, 5> parameter;
         parameter(1) = fitpar(1);
         parameter(2) = ::exp(-fitpar(2));

         const TPAR fp32 = fitpar(3);
         const TPAR fp42 = fitpar(4);
         const TPAR qdif = fp32 - fp42;
         const TPAR qsum = fp32 + fp42;

         TPAR w2, qdcw;
         const TPAR error_limit2 = TPAR(4.0);
         const TPAR qdif2 = pow2(qdif);
         const TPAR qdiferr2 = trace(3) + trace(4);
         const TPAR fp52  = pow2( fitpar(5) );
         const TPAR fp5err2  = trace(5);

         if( qdif2 < (error_limit2 * qdiferr2) ) // rather round in x to y
         {
            if( (fp52 < (error_limit2 * fp5err2)) &&
                (fp52 < qdif2) ) // and not diagonal
            {
               w2 = TPAR(0.0);
               qdcw = qdif;
            }
            else // but still diagonal
            {
               //w2 = ::atan2(fitpar(5), copysign(0.0, qdif));
               //qdcw = -copysign(fitpar(5), qdif);
               w2 = ::atan2( fitpar(5), qdif );
               qdcw = qdif / ::cos( w2 );
            }
         }
         else
         {
            w2 = ::atan2( fitpar(5), qdif );
            qdcw = qdif / ::cos( w2 );
         }
         //parameter(5) = w2 / TPAR(2.0 * M_PI / 180.0);
         parameter(5) = w2 * TPAR(90.0 / M_PI);

         const TPAR sx = (qsum + qdcw) / TPAR(2.0);
         const TPAR sy = (qsum - qdcw) / TPAR(2.0);
         parameter(3) = (sx >= TPAR(0.0)) ? 
            TPAR(2.0 * ::sqrt(M_LN2 / sx)) :
            -TPAR(2.0 * ::sqrt(-M_LN2 / sx));
         parameter(4) = (sy >= TPAR(0.0)) ? 
            TPAR(2.0 * ::sqrt(M_LN2 / sy)) :
            -TPAR(2.0 * ::sqrt(-M_LN2 / sy));

         return parameter;
      }
      
      inline FVector<TPAR, 5>
      covtoerr(const typename FMatrix<TPAR, 5, 5>::TraceVector& trace,
               const FVector<TPAR, 5>& fitpar)
      {
         //throw LinearAlgebraException("Still missing ...");
         // !!! error propagation is too complicated -> use a trick
         Gaussian<TPAR, TDAT, 5, 2> simplegauss;
         simplegauss.setData(this->data_, this->nan_data_, this->error2_);
         FVector<TPAR, 5> parameter = simplegauss.partofit(fittopar(fitpar, trace)); 
         FMatrix<TPAR, 5, 5> cov;
         FVector<TPAR, 5> dummy;
         simplegauss.marquardtCoefficients(parameter,
                                           TPAR(0.0), cov, dummy);
         GaussJ<TPAR, 5>::eval(cov, dummy);
         //error(1) = trace(1);
         //const TPAR ea22 = pow2( exp(-fitpar(2)) );
         //error(2) = ea22 * trace(2);
         // ...
         //error(3) = trace(3);
         //error(4) = trace(4);
         //error(5) = trace(5);
         return simplegauss.covtoerr(cov.traceVector(), parameter);
      }
      
      inline static MArray<TDAT, 2>&
      fill(const FVector<TPAR, 5>& restrict_ invalue,
	   const int x, const int y,
	   MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 5> fitpar(partofit(invalue));
         const TPAR am(fitpar(2));
         const TPAR bx(fitpar(3)); // xwidth
         const TPAR by(fitpar(4)); // ywidth
         const TPAR bxy(fitpar(5)); // mixed width

         a = fitpar(1) +
            exp( -( (bx * pow2( (indexPosDbl(a, 1) - x) )) +
                    (by * pow2( (indexPosDbl(a, 2) - y) )) +
                    (bxy * (indexPosDbl(a, 1) - x) * (indexPosDbl(a, 2) - y)) +
                    am ) );
         return a;
      }

      inline static MArray<TDAT, 2>&
      fillExp(const FVector<TPAR, 5>& restrict_ invalue,
              const int x, const int y,
              MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 5> fitpar(partofit(invalue));
         const TPAR am(fitpar(2));
         const TPAR bx(fitpar(3)); // xwidth
         const TPAR by(fitpar(4)); // ywidth
         const TPAR bxy(fitpar(5)); // mixed width

         a = -( (bx * pow2( (indexPosDbl(a, 1) - x) )) +
                (by * pow2( (indexPosDbl(a, 2) - y) )) +
                (bxy * (indexPosDbl(a, 1) - x) * (indexPosDbl(a, 2) - y)) +
                am );
         return a;
      }

      inline TPAR marquardtCoefficients(const FVector<TPAR, 5>& restrict_ parameter,
                                        const TPAR chisquare_limit,
                                        FMatrix<TPAR, 5, 5>& restrict_ a,
                                        FVector<TPAR, 5>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( M_LN2 * TPAR(DBL_MAX_EXP - 1));
         const TPAR le_limit( M_LN2 * TPAR(DBL_MIN_EXP - 1));
         //const TPAR ue_limit( 18.0 );
         //const TPAR le_limit( -14.0 );

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         //const TPAR dx(TPAR(2.0) * parameter(3));
         //const TPAR bx(pow2(parameter(3))); // xwidth
         //const TPAR dy(TPAR(2.0) * parameter(4));
         //const TPAR by(pow2(parameter(4))); // ywidth
         const TPAR bx(parameter(3)); // xwidth
         const TPAR by(parameter(4)); // ywidth
         const TPAR bxy(parameter(5)); // mixed width

         tMatClearHesse<TPAR, 5>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 5> derivative;
         derivative(1) = TPAR(1.0);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         typename MArray<TDAT, 2>::IndexIterator index_i = this->data_.indexBegin();
         typename MArray<TDAT, 2>::const_iterator rdata_i = rdata_.begin();
         
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();
         typename MArray<TDAT, 2>::const_iterator rerror2_i = rerror2_.begin();

         TPAR chisquare = TPAR(0);

         while(data_i != rdata_i)
         {
            const bool have_data = (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
            const bool have_rdata = (*rdata_i != this->nan_data_) && (*rerror2_i > TDAT(0));
            if( have_data || have_rdata )
            {
               const TPAR ydif  = TPAR(index_i(2));
               const TPAR y2    = ydif * ydif;
               const TPAR xdif = TPAR(index_i(1));
               const TPAR x2 = xdif * xdif;
               const TPAR xy = xdif * ydif;

               const TPAR erta = - ( x2 * bx + y2 * by + xy * bxy + am );
               const TPAR eberta = (erta < le_limit) ?
                  TPAR(0.0) :
                  ( (erta > ue_limit) ?
                    ::exp(ue_limit) : ::exp(erta) );
               const TPAR meberta = - eberta;
               derivative(2) = meberta;
               //derivative(3) = meberta * x2 * dx;
               //derivative(4) = meberta * y2 * dy;
               derivative(3) = meberta * x2;
               derivative(4) = meberta * y2;
               derivative(5) = meberta * xy;
               if(have_data)
               {
                  const TPAR df = TPAR(*data_i - TDAT(eberta + sf));
                  const TPAR sig2 = TPAR(*error2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0.0)) )
                     return chisquare;
                  tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
               }
               if(have_rdata)
               {
                  const TPAR df = TPAR(*rdata_i - TDAT(eberta + sf));
                  const TPAR sig2 = TPAR(*rerror2_i);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0.0)) )
                     return chisquare;
                  tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
               }
            }
            ++data_i;
            ++index_i;
            ++error2_i;
            ++rdata_i;
            ++rerror2_i;
         }
         {
            const bool have_data =
               (*data_i != this->nan_data_) && (*error2_i > TDAT(0));
            if( have_data )
            {
               TPAR erta = -am;
               if(erta > ue_limit)
                  erta = ue_limit;
               else if(erta < le_limit)
                  erta = le_limit;
               const TPAR eberta  = ::exp(erta);
               const TPAR meberta = -eberta;
               //derivative(2) = meberta;
               //derivative(3) = TPAR(0);
               //derivative(4) = TPAR(0);
               //derivative(5) = TPAR(0);
               const TPAR df = TPAR(*data_i - TDAT(eberta + sf));
               const TPAR sig2 = TPAR(*error2_i);
               chisquare += df * df / sig2;
               //tMatHesse<TPAR, 5>::eval(derivative, df, sig2, a, b);
               // means:
               //for(int l=1; l <= 5; ++l)
               //{
               //const TPAR wt = derivative(l) / sig2;
               //for(int k=1; k <= l; ++k)
               //a(l, k) += wt * derivative(k);
               //b(l) += df * wt;
               //}
               // so only next are left
               a(1,1) += TPAR(1.0) / sig2;
               const TPAR ebesig = meberta / sig2;
               a(2,1) += ebesig;
               a(2,2) += meberta * ebesig;
               b(1) += df / sig2;
               b(2) += df * ebesig;
            }
         }

         // fill up rest of matrix
         tMatFillHesse<TPAR, 5>::eval(a);
         return chisquare;
      }

   private:
      MArray<TDAT, 2> rdata_;
      MArray<TDAT, 2> rerror2_;
};

//! Approximation of a Moffat function via Marquardt-Levenberg algorithm.
template<class TPAR, class TDAT, int NPAR, int NDIM>
class Moffat : public MRQFunction<TPAR, TDAT, NPAR, NDIM>
{ };

// polynomial, radians version
// 10-parameter Moffat, i.e. Moffat on tilted surface

// f(x, y) = C + m_x * x + m_y * y + 
//               A * ( 1.0 + {x_w * [(x-x_0)*cos(a) + (y-y_0)*sin(a)]}^2 +
//                           {y_w * [(y-y_0)*cos(a) - (x-x_0)*sin(a)]}^2  )^ beta
// -> i.e. angular version
// f(x, y) = a1 + a9 * x + a10 * y +
//                a2 * ( 1.0 + {a3 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)]}^2 +
//                             {a4 * [(y-a6)*cos(a7) + (x-a5)*sin(a7)]}^2  )^ -a8
//
// Hesse
// df / da1 = 1
// df / da2 = pow(..., -a8)
// df / da3 = -2.0 * a2 * a8 * a3 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)]^2 *
//                   pow(..., -(a8 + 1.0)
// df / da4 = -2.0 * a2 * a8 * a4 * [(y-a6)*cos(a7) + (x-a5)*sin(a7)]^2 *
//                   pow(..., -(a8 + 1.0)
// df / da5 = -2.0 * a2 * a8 * ( -a3^2 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)] * cos(a7) +
//                                a4^2 * [(y-a6)*cos(a7) + (x-a5)*sin(a7)] * sin(a7)  ) *
//                  pow(..., -(a8 + 1.0)
// df / da6 = -2.0 * a2 * a8 * ( -a3^2 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)] * sin(a7) -
//                                a4^2 * [(y-a6)*cos(a7) + (x-a5)*sin(a7)] * cos(a7)  ) *
//                  pow(..., -(a8 + 1.0) 
// df / da7 = -2.0 * a2 * a8 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)] * [(y-a6)*cos(a7) + (x-a5)*sin(a7)] *
//                   (a3^2 - a4^2 ) * pow(..., -(a8 + 1.0)
// df / da8 = -a2 * log( 1.0 + {a3 * [(x-a5)*cos(a7) + (y-a6)*sin(a7)]}^2 +
//                             {a4 * [(y-a6)*cos(a7) + (x-a5)*sin(a7)]}^2  ) * pow(..., -(a8 + 1.0)
// df / da9 = x
// df / da10 = y

template<class TPAR, class TDAT>
class Moffat<TPAR, TDAT, 10, 2> : public MRQFunction<TPAR, TDAT, 10, 2>
{
   public:

      inline Moffat() { }

      inline static void fill(const FVector<TPAR, 10>& restrict_ invalue,
                              MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 10> fitpar(Moffat<TPAR, TDAT, 10, 2>::partofit(invalue));
         const TPAR cw  = ::cos( fitpar(7) );
         const TPAR sw  = ::sin( fitpar(7) );
         MArray<TPAR, 2> b(a.shape()), c(a.shape());
         b = 1.0 + pow2( fitpar(3) * ((indexPosDbl(a, 1) - fitpar(5)) * cw +
                                      (indexPosDbl(a, 2) - fitpar(6)) * sw )) +
            pow2( fitpar(4) * ((indexPosDbl(a, 2) - fitpar(6)) * cw -
                               (indexPosDbl(a, 1) - fitpar(5)) * sw ));
         
         c = (fitpar(8) + 1.0) * log10( b );
         a = merge( c > 100.0,
                    fitpar(1) + fitpar(9) * indexPosDbl(c, 1) + fitpar(10) * indexPosDbl(c, 2),
                    merge( c < -100.0, -1.0e100,
                           fitpar(1) + fitpar(9) * indexPosDbl(c, 1) + fitpar(10) * indexPosDbl(c, 2) +
                           fitpar(2) * pow(b, -fitpar(8)) ) );
      }
      
      TPAR marquardtCoefficients(const FVector<TPAR, 10>& restrict_ parameter,
                                 const TPAR chisquare_limit,
                                 FMatrix<TPAR, 10, 10>& restrict_ a,
                                 FVector<TPAR, 10>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( 100.0 );
         const TPAR le_limit( -100.0 );

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR bx(parameter(3)); // xwidth
         const TPAR bx2 = bx * bx;
         const TPAR by(parameter(4)); // ywidth
         const TPAR by2 = by * by;
         //const TPAR bymbx(by - bx);
         const TPAR cw(::cos(parameter(7))); // cos of angle
         const TPAR sw(::sin(parameter(7))); // sin of angle
         const TPAR x0(parameter(5)); // xposition
         const TPAR y0(parameter(6)); // yposition
         const TPAR beta(parameter(8)); // exponent
         const TPAR bp1(beta+1.0); // beta plus 1
         const TPAR mx(parameter(9)); // x-tilt
         const TPAR my(parameter(10)); // y-tilt
         const TPAR mtwoampbeta = -2.0 * am * beta;

         tMatClearHesse<TPAR, 10>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 10> derivative;
         derivative(1) = TPAR(1);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();

         TPAR chisquare = TPAR(0);

         for(int yy = this->data_.minIndex(2); yy <= this->data_.maxIndex(2); ++yy)
         {
            derivative(10) = TPAR(yy);
            const TPAR fy = sf + my * derivative(10);
            const TPAR ydif = derivative(10) - y0;
            const TPAR ysw = ydif * sw;
            const TPAR ycw = ydif * cw;
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               const TDAT dat = *data_i;
               const TDAT er2 = *error2_i;
               if( (dat != this->nan_data_) && (er2 > TDAT(0)) )
               {
                  derivative(9) = TPAR(xx);
                  TPAR f = fy + mx * derivative(9);
                  const TPAR xdif = derivative(9) - x0;
                  const TPAR t1 = xdif * cw + ysw;
                  const TPAR t2 = ycw - xdif * sw;
                  const TPAR t1s = t1 * t1;
                  const TPAR t2s = t2 * t2;
                  const TPAR ert = 1.0 + (bx2 * t1s + by2 * t2s);
                  const TPAR lg = bp1 * ::log10(ert);
                  if(lg > ue_limit){
                     derivative(8) = TPAR(0);
                     derivative(7) = TPAR(0);
                     derivative(6) = TPAR(0);
                     derivative(5) = TPAR(0);
                     derivative(4) = TPAR(0);
                     derivative(3) = TPAR(0);
                     derivative(2) = TPAR(0);                     
                  }else if(lg < le_limit){
                     throw DivergenceException("Divergent exponent in Moffat within Marquardt-Levenberg");
                  }else{
                     const TPAR inv = ::pow( ert, -bp1);
                     const TPAR mtwoampbeta_inv = mtwoampbeta * inv;
                     derivative(3) = mtwoampbeta_inv * bx * t1s;
                     derivative(4) = mtwoampbeta_inv * by * t2s;
                     derivative(5) = mtwoampbeta_inv * ( (by2 * t2 * sw) - (bx2 * t1 * cw) );
                     derivative(6) = mtwoampbeta_inv * (0.0 - (by2 * t2 * cw) - (bx2 * t1 * sw) );
                     derivative(7) = mtwoampbeta_inv * t1 * t2 * (bx2 - by2);
                     const TPAR est = ::pow( ert, -beta);
                     derivative(2) = est;
                     const TPAR amest = am * est;
                     f += amest;
                     derivative(8) = (0.0 - amest) * ::log( ert );
                  }
                  const TPAR df = TPAR(dat - TDAT(f));
                  const TPAR sig2 = TPAR(er2);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 10>::eval(derivative, df, sig2, a, b);
               }
               ++data_i;
               ++error2_i;
            }
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 10>::eval(a);
         return chisquare;
      }
};

// 8-parameter is 10-parameter without surface tilt
template<class TPAR, class TDAT>
class Moffat<TPAR, TDAT, 8, 2> : public MRQFunction<TPAR, TDAT, 8, 2>
{
   public:

      inline Moffat() { }

      inline static void fill(const FVector<TPAR, 8>& restrict_ invalue,
                              MArray<TDAT, 2>& a)
      {
         const FVector<TPAR, 8> fitpar(Moffat<TPAR, TDAT, 8, 2>::partofit(invalue));
         const TPAR cw  = ::cos( fitpar(7) );
         const TPAR sw  = ::sin( fitpar(7) );
         MArray<TPAR, 2> b(a.shape()), c(a.shape());
         b = 1.0 + pow2( fitpar(3) * ((indexPosDbl(a, 1) - fitpar(5)) * cw +
                                      (indexPosDbl(a, 2) - fitpar(6)) * sw )) +
            pow2( fitpar(4) * ((indexPosDbl(a, 2) - fitpar(6)) * cw -
                               (indexPosDbl(a, 1) - fitpar(5)) * sw ));
         
         c = (fitpar(8) + 1.0) * log10( b );
         a = merge( c > 100.0,
                    fitpar(1),
                    merge( c < -100.0, -1.0e100,
                           fitpar(1) + fitpar(2) * pow(b, 0.0 - fitpar(8)) ) );
      }
      
      TPAR marquardtCoefficients(const FVector<TPAR, 8>& restrict_ parameter,
                                 const TPAR chisquare_limit,
                                 FMatrix<TPAR, 8, 8>& restrict_ a,
                                 FVector<TPAR, 8>& restrict_ b) const
      {
         // constants
         const TPAR ue_limit( 100.0 );
         const TPAR le_limit( -100.0 );

         const TPAR sf(parameter(1)); // surface
         const TPAR am(parameter(2)); // amplitude
         const TPAR bx(parameter(3)); // xwidth
         const TPAR bx2 = bx * bx;
         const TPAR by(parameter(4)); // ywidth
         const TPAR by2 = by * by;
         //const TPAR bymbx(by - bx);
         const TPAR cw(::cos(parameter(7))); // cos of angle
         const TPAR sw(::sin(parameter(7))); // sin of angle
         const TPAR x0(parameter(5)); // xposition
         const TPAR y0(parameter(6)); // yposition
         const TPAR beta(parameter(8)); // exponent
         const TPAR bp1(beta+1.0); // beta plus 1
         const TPAR mtwoampbeta = -2.0 * am * beta;

         tMatClearHesse<TPAR, 8>::eval(a);
         b = TPAR(0);
         FVector<TPAR, 8> derivative;
         derivative(1) = TPAR(1);

         typename MArray<TDAT, 2>::const_iterator data_i = this->data_.begin();
         typename MArray<TDAT, 2>::const_iterator error2_i = this->error2_.begin();

         TPAR chisquare = TPAR(0);

         for(int yy = this->data_.minIndex(2); yy <= this->data_.maxIndex(2); ++yy)
         {
            const TPAR ydif = TPAR(yy) - y0;
            const TPAR ysw = ydif * sw;
            const TPAR ycw = ydif * cw;
            for(int xx = this->data_.minIndex(1); xx <= this->data_.maxIndex(1); ++xx)
            {
               const TDAT dat = *data_i;
               const TDAT er2 = *error2_i;
               if( (dat != this->nan_data_) && (er2 > TDAT(0)) )
               {
                  TPAR f = sf;
                  const TPAR xdif = TPAR(xx) - x0;
                  const TPAR t1 = xdif * cw + ysw;
                  const TPAR t2 = ycw - xdif * sw;
                  const TPAR t1s = t1 * t1;
                  const TPAR t2s = t2 * t2;
                  const TPAR ert = 1.0 + (bx2 * t1s + by2 * t2s);
                  const TPAR lg = bp1 * ::log10(ert);
                  if(lg > ue_limit){
                     derivative(8) = TPAR(0);
                     derivative(7) = TPAR(0);
                     derivative(6) = TPAR(0);
                     derivative(5) = TPAR(0);
                     derivative(4) = TPAR(0);
                     derivative(3) = TPAR(0);
                     derivative(2) = TPAR(0);                     
                  }else if(lg < le_limit){
                     throw DivergenceException("Divergent exponent in Moffat within Marquardt-Levenberg");
                  }else{
                     const TPAR inv = ::pow( ert, 0.0 - bp1);
                     const TPAR mtwoampbeta_inv = mtwoampbeta * inv;
                     derivative(3) = mtwoampbeta_inv * bx * t1s;
                     derivative(4) = mtwoampbeta_inv * by * t2s;
                     derivative(5) = mtwoampbeta_inv * ( (by2 * t2 * sw) - (bx2 * t1 * cw) );
                     derivative(6) = mtwoampbeta_inv * ( 0.0 - (by2 * t2 * cw) - (bx2 * t1 * sw) );
                     derivative(7) = mtwoampbeta_inv * t1 * t2 * (bx2 - by2);
                     const TPAR est = ::pow( ert, 0.0 - beta);
                     derivative(2) = est;
                     const TPAR amest = am * est;
                     f += amest;
                     derivative(8) = (0.0 - amest) * ::log( ert );
                  }
                  const TPAR df = TPAR(dat - TDAT(f));
                  const TPAR sig2 = TPAR(er2);
                  chisquare += df * df / sig2;
                  if( (chisquare > chisquare_limit) &&
                      (chisquare_limit != TPAR(0)) )
                     return chisquare;
                  tMatHesse<TPAR, 8>::eval(derivative, df, sig2, a, b);
               }
               ++data_i;
               ++error2_i;
            }
         }
         // fill up rest of matrix
         tMatFillHesse<TPAR, 8>::eval(a);
         return chisquare;
      }
};

//! Approximation of a Moffat function via Marquardt-Levenberg algorithm.
// version with rotation angle in degree and width parameters in terms of FWHM
template<class TPAR, class TDAT, int NPAR, int NDIM>
class DegMoffat : public Moffat<TPAR, TDAT, NPAR, NDIM>
{ };

template<class TPAR, class TDAT>
class DegMoffat<TPAR, TDAT, 10, 2> : public Moffat<TPAR, TDAT, 10, 2>
{
   public:
      inline static FVector<TPAR, 10>
      partofit(const FVector<TPAR, 10>& parameter)
      {
         FVector<TPAR, 10> fitpar;
         fitpar(10) = parameter(10); // y-tilt
         fitpar(9) = parameter(9); // x-tilt
         fitpar(8) = parameter(8); // -beta start
         const TPAR fwhmscale = 2.0 * ::sqrt( ::pow( 2.0, 1.0 / fitpar(8) ) - 1.0 );
         fitpar(7) = parameter(7) * M_PI / 180.0; // start angle deg2rad
         fitpar(7) = ::fmod( fitpar(7), M_PI );
         if(fitpar(7) < 0.0) fitpar(7) += M_PI;
         fitpar(6) = parameter(6); // start y
         fitpar(5) = parameter(5); // start x
         fitpar(4) = fwhmscale / parameter(4);
         fitpar(3) = fwhmscale / parameter(3);
         fitpar(2) = parameter(2); // start amplitude
         fitpar(1) = parameter(1); // start constant surface
         return fitpar;
      }

      inline static FVector<TPAR, 10>
      fittopar(const FVector<TPAR, 10>& fitpar,
               const typename FMatrix<TPAR, 10, 10>::TraceVector& trace)
      {
         FVector<TPAR, 10> parameter;
         parameter(10) = fitpar(10);
         parameter(9) = fitpar(9);
         parameter(8) = fitpar(8);
         const TPAR fwhmscale = 2.0 * ::sqrt( ::pow( 2.0, 1.0 / fitpar(8) ) - 1.0 );
         parameter(7) = fitpar(7) * 180.0 / M_PI; // start angle deg2rad
         parameter(7) = ::fmod( parameter(7), 180.0 );
         if(parameter(7) < 0.0) parameter(7) += 180.0;
         parameter(6) = fitpar(6);
         parameter(5) = fitpar(5);
         parameter(4) = fwhmscale / fitpar(4);
         parameter(3) = fwhmscale / fitpar(3);
         parameter(2) = fitpar(2);
         parameter(1) = fitpar(1);
         return parameter;
      }      
      inline FVector<TPAR, 10>
      covtoerr(const typename FMatrix<TPAR, 10, 10>::TraceVector& trace,
               const FVector<TPAR, 10>& fitpar)
      {
         throw LinearAlgebraException("error conversion still missing ...");
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         FVector<TPAR, 10> error;
         //error(1) = trace(1);
         //const TPAR ea22 = pow2( exp(-fitpar(2)) );
         //error(2) = ea22 * trace(2);
         // ...
         //error(3) = trace(3);
         //error(4) = trace(4);
         //error(5) = trace(5);
         //error(6) = trace(6);
         //error(7) = trace(7);
         // ...
         error = trace;
         return error;
      }
};

template<class TPAR, class TDAT>
class DegMoffat<TPAR, TDAT, 8, 2> : public Moffat<TPAR, TDAT, 8, 2>
{
   public:
      inline static FVector<TPAR, 8>
      partofit(const FVector<TPAR, 8>& parameter)
      {
         FVector<TPAR, 8> fitpar;
         fitpar(8) = parameter(8); // -beta start
         const TPAR fwhmscale = 2.0 * ::sqrt( ::pow( 2.0, 1.0 / fitpar(8) ) - 1.0 );
         fitpar(7) = parameter(7) * M_PI / 180.0; // start angle deg2rad
         fitpar(7) = ::fmod( fitpar(7), M_PI );
         if(fitpar(7) < 0.0) fitpar(7) += M_PI;
         fitpar(6) = parameter(6); // start y
         fitpar(5) = parameter(5); // start x
         fitpar(4) = fwhmscale / parameter(4);
         fitpar(3) = fwhmscale / parameter(3);
         fitpar(2) = parameter(2); // start amplitude
         fitpar(1) = parameter(1); // start constant surface
         return fitpar;
      }

      inline static FVector<TPAR, 8>
      fittopar(const FVector<TPAR, 8>& fitpar,
               const typename FMatrix<TPAR, 8, 8>::TraceVector& trace)
      {
         FVector<TPAR, 8> parameter;
         parameter(8) = fitpar(8);
         const TPAR fwhmscale = 2.0 * ::sqrt( ::pow( 2.0, 1.0 / fitpar(8) ) - 1.0 );
         parameter(7) = fitpar(7) * 180.0 / M_PI; // start angle deg2rad
         parameter(7) = ::fmod( parameter(7), 180.0 );
         if(parameter(7) < 0.0) parameter(7) += 180.0;
         parameter(6) = fitpar(6);
         parameter(5) = fitpar(5);
         parameter(4) = fwhmscale / fitpar(4);
         parameter(3) = fwhmscale / fitpar(3);
         parameter(2) = fitpar(2);
         parameter(1) = fitpar(1);
         return parameter;
      }      
      inline FVector<TPAR, 8>
      covtoerr(const typename FMatrix<TPAR, 8, 8>::TraceVector& trace,
               const FVector<TPAR, 8>& fitpar)
      {
         throw LinearAlgebraException("error conversion still missing ...");
         // error(i) = sqrt( sum_j(pow2(d_fitpar(j) / d_trace(j)) * pow2(fitpar(j))) );
         FVector<TPAR, 8> error;         
         //error(1) = trace(1);
         //const TPAR ea22 = pow2( exp(-fitpar(2)) );
         //error(2) = ea22 * trace(2);
         // ...
         //error(3) = trace(3);
         //error(4) = trace(4);
         //error(5) = trace(5);
         //error(6) = trace(6);
         //error(7) = trace(7);
         // ...
         error = trace;
         return error;
      }
};


//@}

#endif
