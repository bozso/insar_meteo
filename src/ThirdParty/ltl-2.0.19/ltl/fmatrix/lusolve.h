/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: lusolve.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_LUSOLVE__
#define __LTL_LUSOLVE__

#include <ltl/fmatrix.h>
#include <ltl/misc/exceptions.h>
#include <cmath> // fabs()

namespace ltl {

#define ZERO 0


/*! \ingroup lusolve
 */
//@{
/*!
 *   Solve the linear system of equations A x = b using LU decomposition.
 */
template <typename T, int N>
class LUDecomposition
{
   public:
      /*!
       *   Solve the linear system of equations A x = b using LU decomposition.
       *   The solution vector x is returned.
       */
      static FVector<T,N> solve( FMatrix<T,N,N> A, FVector<T,N> b )
      {
         FVector<int,N> Index;
         ludcmp( A, Index );
         lubksb( A, Index, b );
         return b;
      }

      /*!
       *   Solve the linear system of equations A x = b using LU decomposition
       *   for each column of the matrix b. The result vectors x are returned as
       *   columns of a matrix X.
       */
      static FMatrix<T,N,N> solve( FMatrix<T,N,N> A, FMatrix<T,N,N>& B )
      {
         FMatrix<T,N,N> X;
         for (int i=1; i<=N; ++i)
         {
            FVector<T,N> b;
            b = B.col(i);
            X.col(i) = solve( A, b );
         }
         return X;
      }

      /*!
       *   Invert the matrix A using LU decomposition.
       */
      static FMatrix<T,N,N> invert( FMatrix<T,N,N> A )
      {
         FMatrix<T,N,N> B;
         B = 0;
         B.traceVector() = 1;

         return solve(A,B);
      }

   protected:
      /*
       * The function
       *       ludcmp()
       * takes as input a LTL matrix FMatrix<T,N,N> A of dimension N and
       * replaces it by the LU decomposition of a rowwise permutation of
       * itself. The results is stored in A in the form given by
       * eq. (2.3.14) in "Numerical Recipe", sect. 2.3, page 45. The vector
       * FVector<int,N> Index records the row permutation effected by the
       * partial pivoting; This
       * routine is used in combination with the template function lubksb()
       * to solve linear equations or invert a matrix.
       */
      static void ludcmp( FMatrix<T,N,N>& A, FVector<int,N>& Index );

      /*
       * The template function
       *             lubksb()
       * solves the set of linear equations
       *           A x = b
       * of dimension N. FMatrix<T,N,N> A is input, as its LU decomposition
       * determined by the template function ludcmp(),
       * FVector<int,N> Index is input as the permutation vector
       * returned by ludcmp(). FVector<T,N> B is input as the
       * right-hand side vector B,
       * The solution X is returned in B(). The input data A()
       * and Index() are not modified. This routine take into
       * account the possibility that B() will begin with many
       * zero elements, so it is efficient for use in matrix
       * inversion.
       */
      static void lubksb(const FMatrix<T,N,N>& A, const FVector<int,N>& Index, FVector<T,N>& B);
};


template<typename T, int N>
void LUDecomposition<T,N>::ludcmp( FMatrix<T,N,N>& A, FVector<int,N>& Index )
{
   FVector<T,N> vv;

   for (int i = 1; i <= N; i++)
   { // loop over rows to get scaling information
      T big = ZERO;
      for (int j = 1; j <= N; j++)
      {
         T f = ::fabs(A(i, j));
         if (f > big)
            big = f;
      }
      if (big == ZERO)
         throw SingularMatrixException("Singular matrix in LU decomposition");
      vv(i) = 1.0/big; // save scaling */
   }

   for (int j = 1; j <= N; j++)
   {
      int imax=1;
      // loop over columns of Crout's method
      for (int i = 1; i < j; i++)
      { // not i = j
         T sum = A(i, j);
         for (int k = 1; k < i; k++)
            sum -= A(i, k) * A(k, j);
         A(i, j) = sum;
      }
      T big = ZERO; // initialization for search for largest pivot element
      for (int i = j; i <= N; i++)
      {
         T sum = A(i, j);
         for (int k = 1; k < j; k++)
         {
            sum -= A(i, k) * A(k, j);
         }
         A(i, j) = sum;
         T f = vv(i)*::fabs(sum);
         if (f >= big)
         {
            big = f;
            imax = i;
         }
      } // end i-loop
      if (j != imax)
      { // do we need to interchange rows ?
         for (int k = 1; k <= N; k++)
         { // yes
            T f = A(imax, k);
            A(imax, k) = A(j,k);
            A(j, k) = f;
         }
         vv(imax) = vv(j); // also interchange scaling factor
      }
      Index(j) = imax;
      if (::fabs(A(j, j)) <= ZERO)
         A(j, j) = ZERO;

      /*
       * if the pivot element is zero the matrix is singular
       * (at least to the precision of the algorithm). For
       * some application of singular matrices, it is desirable
       * to substitute ZERO for zero,
       */

      if (j < N)
      { // divide by pivot element
         T f = 1.0/A(j, j);
         for (int i = j+1; i <= N; i++)
            A(i, j) *= f;
      }
   } // end j-loop over columns
}

template<typename T, int N>
void LUDecomposition<T,N>::lubksb(const FMatrix<T,N,N>& A, const FVector<int,N>& Index, FVector<T,N>& B)
{
   for (int i=1, ii=0; i <= N; i++)
   {
      int ip = Index(i);
      T sum = B(ip);
      B(ip) = B(i);
      if (ii != 0)
      {
         for (int j = ii - 1; j < i; j++)
            sum -= A(i, j) * B(j);
      }
      else if (sum != 0.0)
         ii = i + 1;
      B(i) = sum;
   }

   for (int i = N; i > 0; --i)
   {
      T sum = B(i);
      for (int j = i+1; j <= N; j++)
         sum -= A(i, j) * B(j);
      B(i) = sum/A(i,i);
   }
}
//@}
#undef ZERO

}

#endif
