/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: polynomfit.h 499 2011-12-16 23:41:55Z drory $
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

#ifndef __LTL_POLYNOMFIT_H__
#define __LTL_POLYNOMFIT_H__

#include <ltl/fmatrix/gaussj.h>
#include <cstdio>

namespace ltl {

/*! \addtogroup linlsq

*/
//@{

//! Class to fit a NPAR polynome to an NDIM ltl::MArray
/*!
Template parameter list:
\li TPAR type of fit result (polynome parameters)
\li TDAT type of input MArrays values
\li ORDER highest exponent in polynome
\li EXT: true  = do any mixed terms for multi dim 
         false = restrict mixed terms to sum( exponents ) <= ORDER
\li NDIM dimensions of input MArrays
*/
template<class TPAR, class TDAT, int ORDER, bool EXT, int NDIM>
class PolynomFit
{ };

//! Specialisation for NDIM == 1.
template<class TPAR, class TDAT, int ORDER, bool EXT>
class PolynomFit<TPAR, TDAT, ORDER, EXT, 1>
{
   protected:
      enum { NPAR = (ORDER + 1) };

      inline static FMatrix<TPAR, NPAR, NPAR>
      polynomMatrix(const FVector<TPAR, NPAR>& restrict_ b, const TPAR c)
      {
         FMatrix<TPAR, NPAR, NPAR> a;
         for(int j = 1; j <= NPAR; ++j)
         {
            const TPAR tmp = c * b(j);
            for(int i = 1; i <= j; ++i)
               a(j, i) = b(i) * tmp;
         }
         // next can be avoided by doing only after sum loop in eval
//          for(int j = 2; j <= NPAR; ++j)
//          {
//             for(int i = 1; i < j; ++i)
//                a(i, j) = a(j, i);
//          }
         return a;
       }

   public:

      inline static FVector<TPAR, NPAR> polynomVector(const TPAR x)
      {
         FVector<TPAR, NPAR > xp;
         TPAR xtmp = TPAR(1.0);
         xp(1) = xtmp;
         for(int i = 2; i <= NPAR; ++i)
         {
            xtmp *= x;
            xp(i) = xtmp;
         }
         return xp;
      }

      inline static FVector<TPAR, NPAR>
      eval(const MArray<TDAT, 1>& data,
           const MArray<TDAT, 1>& error2)
      {
         typename MArray<TDAT, 1>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 1>::const_iterator d = data.begin();
         typename MArray<TDAT, 1>::const_iterator e = error2.begin();

         FMatrix<TPAR, NPAR, NPAR> A = TPAR(0.0);
         FVector<TPAR, NPAR> B = TPAR(0.0);
         
         while(!d.done())
         {
            const TDAT e_val = *e;
            if(e_val > TDAT(0.0))
            {
               const TPAR em2 = TPAR(1.0) / TPAR(e_val);
               const TPAR dem2 = TPAR(*d) * em2;
               const FVector<TPAR, NPAR> b_i =
                  PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::polynomVector( TPAR(i(1)) );
               B += dem2 * b_i;
               A += PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::polynomMatrix(b_i, em2);
            }
            ++i;
            ++d;
            ++e;
         }
         // instead of complete filling in polynomMatrix ->
         for(int j = 2; j <= NPAR; ++j)
         {
            for(int i = 1; i < j; ++i)
               A(i, j) = A(j, i);
         }
         return GaussJ<TPAR, NPAR>::solve(A, B);
      }

      inline static void
      fill(const FVector<TPAR, NPAR>& x,
           MArray<TDAT, 1>& data)
      {
         typename MArray<TDAT, 1>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 1>::iterator d = data.begin();
         while(!d.done())
         {
            const FVector<TPAR, NPAR> b = 
               PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::polynomVector( TPAR(i(1)) );
            *d = TDAT(dot(x, b));
            ++i;
            ++d;
         }
      }

      static string
      toString(const FVector<TPAR, NPAR>& b)
      {
         char buf[1024];
         string comment = "f(x) =";
         for(int i = 0; i <= ORDER; ++i)
         {
            snprintf(buf, sizeof(buf), " %+g", b(i+1));
            comment += buf;
            string x;
            switch(i){
               case 0: x = ""; break;
               case 1: x = " x"; break;
               default: snprintf(buf, sizeof(buf), " x^%i", i); x += string(buf); break;
            }
            comment += x;
         }
         return comment;
      }

      inline static MArray<TDAT, 1>
      fit(const MArray<TDAT, 1>& data,
          const MArray<TDAT, 1>& error2)
      {
         MArray<TDAT, 1> fitarray(data.shape());
         PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::fill(
            PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::eval(data, error2),
            fitarray);
         return fitarray;
      }

      inline static MArray<TDAT, 1>
      fit(const MArray<TDAT, 1>& data,
          const MArray<TDAT, 1>& error2,
          string& comment)
      {
         MArray<TDAT, 1> fitarray(data.shape());
         const FVector<TPAR, NPAR> x =
            PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::eval(data, error2);
         PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::fill(x, fitarray);
         comment = PolynomFit<TPAR, TDAT, ORDER, EXT, 1>::toString(x);
         return fitarray;
      }
};

//! Specialisation for NDIM == 2 and restrict mixed terms to sum( exponents ) <= ORDER.
template<class TPAR, class TDAT, int ORDER>
class PolynomFit<TPAR, TDAT, ORDER, false, 2>
{
   protected:
      enum { NPAR = ((ORDER + 1) * (ORDER + 2)) / 2 };

      inline static FVector<TPAR, NPAR> polynomVector(const TPAR x, const TPAR y)
      {
         FVector<TPAR, (ORDER + 1) > xp =
            PolynomFit<TPAR, TDAT, ORDER, false, 1>::polynomVector( x );
            
         FVector<TPAR, (ORDER + 1) > yp =
            PolynomFit<TPAR, TDAT, ORDER, false, 1>::polynomVector( y );
         FVector<TPAR, NPAR> b;
         for(int j = 1, k = 1; j <= (ORDER + 1); ++j)
         {
            const TPAR yp_j = yp(j);
            for(int i = 1; (i + j) <= (ORDER + 2); ++i, ++k)
               b(k) = xp(i) * yp_j;
         }
         return b;
      }

      inline static FMatrix<TPAR, NPAR, NPAR>
      polynomMatrix(const FVector<TPAR, NPAR>& restrict_ b, const TPAR c)
      {
         FMatrix<TPAR, NPAR, NPAR> a;
         for(int j = 1; j <= NPAR; ++j)
         {
            const TPAR tmp = c * b(j);
            for(int i = 1; i <= j; ++i)
               a(j, i) = b(i) * tmp;
         }
         // next can be avoided by doing only after sum loop in eval
//          for(int j = 2; j <= NPAR; ++j)
//          {
//             for(int i = 1; i < j; ++i)
//                a(i, j) = a(j, i);
//          }
         return a;
       }

   public:

      inline static FVector<TPAR, NPAR>
      eval(const MArray<TDAT, 2>& data,
           const MArray<TDAT, 2>& error2)
      {
         typename MArray<TDAT, 2>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 2>::const_iterator d = data.begin();
         typename MArray<TDAT, 2>::const_iterator e = error2.begin();

         FMatrix<TPAR, NPAR, NPAR> A = TPAR(0.0);
         FVector<TPAR, NPAR> B = TPAR(0.0);
         
         while(!d.done())
         {
            const TDAT e_val = *e;
            if(e_val > TDAT(0.0))
            {
               const TPAR em2 = TPAR(1.0) / TPAR(e_val);
               const TPAR dem2 = TPAR(*d) * em2;
               const FVector<TPAR, NPAR> b_i =
                  PolynomFit<TPAR, TDAT, ORDER, false, 2>::polynomVector( TPAR(i(1)),
                                                                          TPAR(i(2)) );
               B += dem2 * b_i;
               A += PolynomFit<TPAR, TDAT, ORDER, false, 2>::polynomMatrix(b_i, em2);
            }
            ++i;
            ++d;
            ++e;
         }
         // instead of complete filling in polynomMatrix ->
         for(int j = 2; j <= NPAR; ++j)
         {
            for(int i = 1; i < j; ++i)
               A(i, j) = A(j, i);
         }
         return GaussJ<TPAR, NPAR>::solve(A, B);
      }

      inline static void
      fill(const FVector<TPAR, NPAR>& x,
           MArray<TDAT, 2>& data)
      {
         typename MArray<TDAT, 2>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 2>::iterator d = data.begin();
         while(!d.done())
         {
            const FVector<TPAR, NPAR> b = 
               PolynomFit<TPAR, TDAT, ORDER, false, 2>::polynomVector( TPAR(i(1)),
                                                                      TPAR(i(2)) );
            *d = TDAT(dot(x, b));
            ++i;
            ++d;
         }
      }

      static string
      toString(const FVector<TPAR, NPAR>& b)
      {
         char buf[1024];
         string comment = "f(x, y) =";
         for(int j = 0, k = 1; j <= ORDER; ++j)
         {
            string y;
            switch(j){
               case 0: y = ""; break;
               case 1: y = " y"; break;
               default: snprintf(buf, sizeof(buf), " y^%i", j); y += string(buf); break;
            }

            for(int i = 0; (i + j) <= ORDER; ++i, ++k)
            {
               snprintf(buf, sizeof(buf), " %+g", b(k));
               comment += buf;
               string x;
               switch(i){
                  case 0: x = ""; break;
                  case 1: x = " x"; break;
                  default: snprintf(buf, sizeof(buf)," x^%i", i); x += string(buf); break;
               }
               comment += x + y;
            }
         }
         return comment;
      }

      inline static MArray<TDAT, 2>
      fit(const MArray<TDAT, 2>& data,
          const MArray<TDAT, 2>& error2)
      {
         MArray<TDAT, 2> fitarray(data.shape());
         PolynomFit<TPAR, TDAT, ORDER, false, 2>::fill(
            PolynomFit<TPAR, TDAT, ORDER, false, 2>::eval(data, error2),
            fitarray);
         return fitarray;
      }

      inline static MArray<TDAT, 2>
      fit(const MArray<TDAT, 2>& data,
          const MArray<TDAT, 2>& error2,
          string& comment)
      {
         MArray<TDAT, 2> fitarray(data.shape());
         const FVector<TPAR, NPAR> x =
            PolynomFit<TPAR, TDAT, ORDER, false, 2>::eval(data, error2);
         PolynomFit<TPAR, TDAT, ORDER, false, 2>::fill(x, fitarray);
         comment = PolynomFit<TPAR, TDAT, ORDER, false, 2>::toString(x);
         return fitarray;
      }
};

//! Specialisation for NDIM == 2 and any mixed terms.
template<class TPAR, class TDAT, int ORDER>
class PolynomFit<TPAR, TDAT, ORDER, true, 2>
{
   protected:
      enum { NPAR = (ORDER + 1) * (ORDER + 1) };

      inline static FVector<TPAR, NPAR> polynomVector(const TPAR x, const TPAR y)
      {
         FVector<TPAR, (ORDER + 1) > xp =
            PolynomFit<TPAR, TDAT, ORDER, true, 1>::polynomVector( x );
            
         FVector<TPAR, (ORDER + 1) > yp =
            PolynomFit<TPAR, TDAT, ORDER, true, 1>::polynomVector( y );
         FVector<TPAR, NPAR> b;
         for(int j = 1, k = 1; j <= (ORDER + 1); ++j)
         {
            const TPAR yp_j = yp(j);
            for(int i = 1; i <= (ORDER + 1); ++i, ++k)
               b(k) = xp(i) * yp_j;
         }
         return b;
      }

      inline static FMatrix<TPAR, NPAR, NPAR>
      polynomMatrix(const FVector<TPAR, NPAR>& restrict_ b, const TPAR c)
      {
         FMatrix<TPAR, NPAR, NPAR> a;
         for(int j = 1; j <= NPAR; ++j)
         {
            const TPAR tmp = c * b(j);
            for(int i = 1; i <= j; ++i)
               a(j, i) = b(i) * tmp;
         }
         // next can be avoided by doing only after sum loop in eval
//          for(int j = 2; j <= NPAR; ++j)
//          {
//             for(int i = 1; i < j; ++i)
//                a(i, j) = a(j, i);
//          }
         return a;
       }

   public:

      inline static FVector<TPAR, NPAR>
      eval(const MArray<TDAT, 2>& data,
           const MArray<TDAT, 2>& error2)
      {
         typename MArray<TDAT, 2>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 2>::const_iterator d = data.begin();
         typename MArray<TDAT, 2>::const_iterator e = error2.begin();

         FMatrix<TPAR, NPAR, NPAR> A = TPAR(0.0);
         FVector<TPAR, NPAR> B = TPAR(0.0);
         
         while(!d.done())
         {
            const TDAT e_val = *e;
            if(e_val > TDAT(0.0))
            {
               const TPAR em2 = TPAR(1.0) / TPAR(e_val);
               const TPAR dem2 = TPAR(*d) * em2;
               const FVector<TPAR, NPAR> b_i =
                  PolynomFit<TPAR, TDAT, ORDER, true, 2>::polynomVector( TPAR(i(1)),
                                                                         TPAR(i(2)) );
               B += dem2 * b_i;
               A += PolynomFit<TPAR, TDAT, ORDER, true, 2>::polynomMatrix(b_i, em2);
            }
            ++i;
            ++d;
            ++e;
         }
         // instead of complete filling in polynomMatrix ->
         for(int j = 2; j <= NPAR; ++j)
         {
            for(int i = 1; i < j; ++i)
               A(i, j) = A(j, i);
         }
         return GaussJ<TPAR, NPAR>::solve(A, B);
      }

      inline static void
      fill(const FVector<TPAR, NPAR>& x,
           MArray<TDAT, 2>& data)
      {
         typename MArray<TDAT, 2>::IndexIterator i = data.indexBegin();
         typename MArray<TDAT, 2>::iterator d = data.begin();
         while(!d.done())
         {
            const FVector<TPAR, NPAR> b = 
               PolynomFit<TPAR, TDAT, ORDER, true, 2>::polynomVector( TPAR(i(1)),
                                                                      TPAR(i(2)) );
            *d = TDAT(dot(x, b));
            ++i;
            ++d;
         }
      }

      static string
      toString(const FVector<TPAR, NPAR>& b)
      {
         char buf[1024];
         string comment = "f(x, y) =";
         for(int j = 0, k = 1; j <= ORDER; ++j)
         {
            string y;
            switch(j){
               case 0: y = ""; break;
               case 1: y = " y"; break;
               default: snprintf(buf, sizeof(buf), " y^%i", j); y += string(buf); break;
            }

            for(int i = 0; i <= ORDER; ++i, ++k)
            {
               snprintf(buf, sizeof(buf), " %+g", b(k));
               comment += buf;
               string x;
               switch(i){
                  case 0: x = ""; break;
                  case 1: x = " x"; break;
                  default: snprintf(buf, sizeof(buf), " x^%i", i); x += string(buf); break;
               }
               comment += x + y;
            }
         }
         return comment;
      }

      inline static MArray<TDAT, 2>
      fit(const MArray<TDAT, 2>& data,
          const MArray<TDAT, 2>& error2)
      {
         MArray<TDAT, 2> fitarray(data.shape());
         PolynomFit<TPAR, TDAT, ORDER, true, 2>::fill(
            PolynomFit<TPAR, TDAT, ORDER, true, 2>::eval(data, error2),
            fitarray);
         return fitarray;
      }

      inline static MArray<TDAT, 2>
      fit(const MArray<TDAT, 2>& data,
          const MArray<TDAT, 2>& error2,
          string& comment)
      {
         MArray<TDAT, 2> fitarray(data.shape());
         const FVector<TPAR, NPAR> x =
            PolynomFit<TPAR, TDAT, ORDER, true, 2>::eval(data, error2);
         PolynomFit<TPAR, TDAT, ORDER, true, 2>::fill(x, fitarray);
         comment = PolynomFit<TPAR, TDAT, ORDER, true, 2>::toString(x);
         return fitarray;
      }
};

//@}

}

#endif
