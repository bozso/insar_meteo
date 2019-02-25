/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: gaussj.h 523 2013-05-22 15:21:45Z cag $
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

#ifndef __LTL_GAUSSJ__
#define __LTL_GAUSSJ__

#include <ltl/fmatrix.h>
#include <ltl/misc/exceptions.h>

namespace ltl {

template<class T>
struct tNMatPivot;

template<class T, int N>
class tMatPivLoop;

template<class T, int N, bool unroll>
class tMatPivSplitLoop;

template<class T, int N, int I, int J>
class tNMatPivLoop;

template<class T, int N>
class tMatElimLoop;

template<class T, int N, bool unroll>
class tMatElimSplitLoop;

template<class T, int N, int L>
class tNMatElimLoop;

template<class T, int N>
class tMatRestLoop;

template<class T, int N, bool unroll>
class tMatRestSplitLoop;

template<class T, int N, int L>
class tNMatRestLoop;

/*! \ingroup gaussj
*/
//@{
//! Solve equation via Gauss-Jordan inversion or just do a Gauss-Jordan inversion
template<class T, int N>
class GaussJ
{
   protected:

      static void getPivot(const FMatrix<T, N, N>& a,
                           FVector<int, N>& ipiv, 
                           tNMatPivot<T>& p)
      {
         tMatPivLoop<T, N>::eval(a, ipiv, p);
      }

      static void swapRows(FMatrix<T, N, N>& a,
                           FVector<T, N>& b,
                           tNMatPivot<T>& p)
      {
         if(p.irow != p.icol)
         {
            // swap matrix lines and vector elements
            a.swapRows(p.irow, p.icol);
            const T dum = b(p.irow);
            b(p.irow) = b(p.icol);
            b(p.icol) = dum;
         }
      }

      static void divByPiv(FMatrix<T, N, N>& a,
                           const tNMatPivot<T>& p)
      {
         if(a(p.icol, p.icol) == T(0.0))
            throw SingularMatrixException("Singular matrix in Gauss-Jordan");
         const T pivinv = T(1.0) / a(p.icol, p.icol);
         a(p.icol, p.icol) = T(1.0);
         typename FMatrix<T,N,N>::RowVector r = a.row(p.icol);
         r *= pivinv;
      }

      static void divByPiv(FMatrix<T, N, N>& a,
                           FVector<T, N>& b,
                           const tNMatPivot<T>& p)
      {
         if(a(p.icol, p.icol) == T(0.0))
            throw SingularMatrixException("Singular matrix in Gauss-Jordan");
         const T pivinv = T(1.0) / a(p.icol, p.icol);
         a(p.icol, p.icol) = T(1.0);
         typename FMatrix<T,N,N>::RowVector r = a.row(p.icol);
         r *= pivinv;
         b(p.icol) *= pivinv;
      }

      static void elimRow(FMatrix<T, N, N>& a,
                          const tNMatPivot<T>& p)
      {
         tMatElimLoop<T, N>::eval(a, p);
      }

      static void elimRow(FMatrix<T, N, N>& a,
                          FVector<T, N>& b,
                          const tNMatPivot<T>& p)
      {
         tMatElimLoop<T, N>::eval(a, b, p);
      }
      
   public:

   //! invert Matrix, similar to eval() but without solving a linear equation
      static FMatrix<T, N, N> invert(FMatrix<T, N, N> a)
      {
         //FVector<T,N> b;
         //b = 1;

         FVector<int, N> indxc;
         FVector<int, N> indxr;
         FVector<int, N> ipiv(0);

         for(int i = 1; i <= N; ++i)
         {
            tNMatPivot<T> p;
            getPivot(a, ipiv, p);
            if(p.irow != p.icol) // swap matrix lines
               a.swapRows(p.irow, p.icol);
            indxr(i) = p.irow;
            indxc(i) = p.icol;
            divByPiv(a, p);
            elimRow(a, p);
         }
         // restore matrix order
         tMatRestLoop<T, N>::eval(a, indxr, indxc);
         return a;
      }

      //! Return the solution vector \c x for the equation <tt>A x = b</tt>
      static FVector<T, N> solve(FMatrix<T, N, N> a, FVector<T, N> b)
      {
         FVector<int, N> ipiv(0);

         for(int i = 1; i <= N; ++i)
         {
            tNMatPivot<T> p;
            getPivot(a, ipiv, p);
            swapRows(a, b, p);
            divByPiv(a, b, p);
            elimRow(a, b, p);
         }
         return b;
      }

      //! Solve <tt>A x = B</tt> by Gauss-Jordan elimination. b is replaced by the solution x, A is replaced by its inverse.
      static void eval(FMatrix<T, N, N>& a, FVector<T, N>& b)
      {
         FVector<int, N> indxc;
         FVector<int, N> indxr;
         FVector<int, N> ipiv(0);

         for(int i = 1; i <= N; ++i)
         {
            tNMatPivot<T> p;
            getPivot(a, ipiv, p);
            swapRows(a, b, p);
            indxr(i) = p.irow;
            indxc(i) = p.icol;
            divByPiv(a, b, p);
            elimRow(a, b, p);
         }
         // restore matrix order
         tMatRestLoop<T, N>::eval(a, indxr, indxc);
      }
};
//@}

/// \cond DOXYGEN_IGNORE
template<class T, int N>
class tMatPivLoop
{
   public:
      enum { static_size = 6 * N * N };

      static void eval( const FMatrix<T, N, N>& a, 
                        FVector<int, N>& ipiv,
                        tNMatPivot<T>& p )
      {
         p.piv = 0.0;
         tMatPivSplitLoop<T, N,
            (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
            >::eval(a, ipiv, p);
         if (p.icol == 0)
            throw SingularMatrixException("Could not find pivot");
         ++(ipiv(p.icol));
      }
};

// unroll pivoting or not
template<class T, int N, bool unroll>
class tMatPivSplitLoop
{ };

template<class T, int N>
class tMatPivSplitLoop<T, N, true>
{
   public:
      inline static void eval( const FMatrix<T, N, N>& a, 
                               const FVector<int, N>& ipiv,
                               tNMatPivot<T>& p )
      {
#ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "Evaluating with template loop unrolled ..."<< endl;
#endif         
         tNMatPivLoop<T, N, N, 0>::eval( a, ipiv, p );
      }
};

template<class T, int N>
class tMatPivSplitLoop<T, N, false>
{
   public:
      static void eval( const FMatrix<T, N, N>& a, 
                        const FVector<int, N>& ipiv,
                        tNMatPivot<T>& p )
      {
         for(int j=1; j <= N; ++j)
         {
            if(ipiv(j) != 1)
            {
               for(int k = 1; k <= N; ++k)
               {
                  if(ipiv(k) == 0)
                  {
                     const double is_piv = ::fabs( a(j, k) );
                     if(is_piv >= p.piv)
                     {
                        p.piv = is_piv;
                        p.irow = j;
                        p.icol = k;
                     }
                  }
                  else if(ipiv(k) > 1)
                     throw SingularMatrixException("Singular Matrix in Pivot of Gauss-Jordan");
               }
            }
         }
      }
};

// start the loop with tNMatPivLoop<T, N, N, 0> !!!
// the K loop
template<class T, int N, int J, int K>
class tNMatPivLoop
{
   public:
      static void eval( const FMatrix<T, N, N>& a, 
                        const FVector<int, N>& ipiv,
                        tNMatPivot<T>& p )
      {
         tNMatPivLoop<T, N, J, K-1>::eval(a, ipiv, p);
         if(ipiv(K) == 0)
         {
            const double is_piv = ::fabs( a(J, K));
            if(is_piv >= p.piv)
            {
               p.piv = is_piv;
               p.irow = J;
               p.icol = K;
            }
         }
         else if(ipiv(K) > 1)
            throw SingularMatrixException("Singular Matrix in Pivot of Gauss-Jordan");
      }
};


// end of K loop
template<class T, int N, int J>
class tNMatPivLoop<T, N, J, 1>
{
   public:
      static void eval( const FMatrix<T, N, N>& a, 
                        const FVector<int, N>& ipiv,
                        tNMatPivot<T>& p )
      {
         if(ipiv(1) == 0)
         {
            const double is_piv = ::fabs( a(J, 1));
            if(is_piv >= p.piv)
            {
               p.piv = is_piv;
               p.irow = J;
               p.icol = 1;
            }
         }
         else if(ipiv(1) > 1)
            throw SingularMatrixException("Singular Matrix in Pivot of Gauss-Jordan");
      }
};


// J loop
template<class T, int N, int J>
class tNMatPivLoop<T, N, J, 0>
{
   public:
      static void eval( const FMatrix<T, N, N>& a, 
                        const FVector<int, N>& ipiv,
                        tNMatPivot<T>& p )
      {
         if(ipiv(J) != 1)
            tNMatPivLoop<T, N, J, N>::eval(a, ipiv, p);
         tNMatPivLoop<T, N, J-1, 0>::eval(a, ipiv, p);
      }
};


template<class T, int N>
class tNMatPivLoop<T, N, 0, 0>
{
   public:
      inline static void eval( const FMatrix<T, N, N>& a, 
                               const FVector<int, N>& ipiv,
                               tNMatPivot<T>& p )
      { }
};


template<class T, int N>
class tMatElimLoop
{
   public:
      enum { static_size = N * (2* N + 3)};

      inline static void eval( FMatrix<T, N, N>& a, 
                               const tNMatPivot<T>& p )
      {
         const typename FMatrix<T,N,N>::RowVector r_icol = a.row(p.icol);
         tMatElimSplitLoop<T, N,
            (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
            >::eval(a, p, r_icol);
      }

      inline static void eval( FMatrix<T, N, N>& a, 
                               FVector<T, N>& b,
                               const tNMatPivot<T>& p )
      {
         const typename FMatrix<T,N,N>::RowVector r_icol = a.row(p.icol);
         tMatElimSplitLoop<T, N,
            (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
            >::eval(a, b, p, r_icol);
      }
};

template<class T, int N, bool unroll>
class tMatElimSplitLoop
{ };

template<class T, int N>
class tMatElimSplitLoop<T, N, true>
{
   public:
      inline static void eval( FMatrix<T, N, N>& a, 
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         tNMatElimLoop<T, N, N>::eval(a, p, r_icol);
      }
      inline static void eval( FMatrix<T, N, N>& a, 
                               FVector<T, N>& b,
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         tNMatElimLoop<T, N, N>::eval(a, b, p, r_icol);
      }
};

template<class T, int N>
class tMatElimSplitLoop<T, N, false>
{
   public:
      static void eval( FMatrix<T, N, N>& a, 
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         for(int ll = 1; ll <= N; ++ll)
         {
            if(ll != p.icol)
            {
               const T factor = a(ll, p.icol);
               a(ll, p.icol) = T(0.0);
               typename FMatrix<T,N,N>::RowVector r_ll = a.row(ll);
               r_ll -= r_icol * factor;
            }
         }
      }
      static void eval( FMatrix<T, N, N>& a, 
                               FVector<T, N>& b,
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         for(int ll = 1; ll <= N; ++ll)
         {
            if(ll != p.icol)
            {
               const T factor = a(ll, p.icol);
               a(ll, p.icol) = T(0.0);
               typename FMatrix<T,N,N>::RowVector r_ll = a.row(ll);
               r_ll -= r_icol * factor;
               b(ll) -= b(p.icol) * factor;
            }
         }
      }
};

template<class T, int N, int L>
class tNMatElimLoop
{
   public:
      static void eval( FMatrix<T, N, N>& a, 
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         tNMatElimLoop<T, N, L-1>::eval(a, p, r_icol);
         if(L != p.icol)
         {
            const T factor = a(L, p.icol);
            a(L, p.icol) = T(0.0);
            typename FMatrix<T,N,N>::RowVector r_L = a.row(L);
            r_L -= r_icol * factor;
         }
      }
      static void eval( FMatrix<T, N, N>& a, 
                               FVector<T, N>& b,
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      {
         tNMatElimLoop<T, N, L-1>::eval(a, b, p, r_icol);
         if(L != p.icol)
         {
            const T factor = a(L, p.icol);
            a(L, p.icol) = T(0.0);
            typename FMatrix<T,N,N>::RowVector r_L = a.row(L);
            r_L -= r_icol * factor;
            b(L) -= b(p.icol) * factor;
         }
      }
      
};

template<class T, int N>
class tNMatElimLoop<T, N, 0>
{
   public:
      inline static void eval( FMatrix<T, N, N>& a, 
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      { }
      inline static void eval( FMatrix<T, N, N>& a, 
                               FVector<T, N>& b,
                               const tNMatPivot<T>& p,
                               const typename FMatrix<T,N,N>::RowVector& r_icol )
      { }
};


template<class T, int N>
class tMatRestLoop
{
   public:
      enum { static_size = 2 * N * N};

      inline static void eval( FMatrix<T, N, N>& a, 
                               const FVector<int, N>& indxr,
                               const FVector<int, N>& indxc )
      {
         tMatRestSplitLoop<T,N,
            (static_size <= LTL_TEMPLATE_LOOP_LIMIT)
            >::eval(a, indxr, indxc);
      }
};

template<class T, int N, bool unroll>
class tMatRestSplitLoop
{ };

template<class T, int N>
class tMatRestSplitLoop<T, N, true>
{
   public:
      inline static void eval( FMatrix<T, N, N>& a, 
                               const FVector<int, N>& indxr,
                               const FVector<int, N>& indxc )
      {
         tNMatRestLoop<T,N,N>::eval(a, indxr, indxc);
      }
};

template<class T, int N>
class tMatRestSplitLoop<T, N, false>
{
   public:
      static void eval( FMatrix<T, N, N>& a, 
                               const FVector<int, N>& indxr,
                               const FVector<int, N>& indxc )
      {
         for(int l = N; l > 0; --l)
         {
            a.swapCols(indxr(l), indxc(l));
         }
      }
};

template<class T, int N, int L>
class tNMatRestLoop
{
   public:
      inline static void eval( FMatrix<T, N, N>& a, 
                               const FVector<int, N>& indxr,
                               const FVector<int, N>& indxc )
      {
         a.swapCols(indxr(L), indxc(L));
         tNMatRestLoop<T, N, L-1>::eval(a, indxr, indxc);
      }
};

template<class T, int N>
class tNMatRestLoop<T, N, 0>
{
   public:
      inline static void eval( FMatrix<T, N, N>& a, 
                               const FVector<int, N>& indxr,
                               const FVector<int, N>& indxc )
      { }
};


template<class T>
struct tNMatPivot
{
   tNMatPivot( int r, int c )
   : irow(r), icol(c) {}
   tNMatPivot()
   : irow(0), icol(0) {}
   
   int irow, icol;
   T piv;
};
/// \endcond
}

#endif
