/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: svdsolve.h 523 2013-05-22 15:21:45Z cag $
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

#ifndef __LTL_FSVD__
#define __LTL_FSVD__

#include <ltl/fmatrix.h>
#include <ltl/misc/exceptions.h>
#include <cmath> // fabs(), sqrt()


namespace ltl {

#define SVD_SIGN(a,b) ((b) >= 0.0 ? ::fabs(a) : -::fabs(a))

/*! \ingroup svdsolve
 */
//@{
/*!
 *   Perform Singular Value Decomposition and solve linear systems.
 *   Adapted from Numerical Recipes.
 *
 *   Example:
 *   \code
 *      FVector<double,4> x(2.);  // target
 *
 *      FMatrix<double,4,4> A;    // the matrix
 *      A = 1., 2.,  3.,  2.,
 *          5., 11., 7.,  13.,
 *          9., 7.,  5.,  1.,
 *          7., 13., 17., 11.;
 *      FVector<double,4> b(dot(A, x));   // construct b
 *
 *      x = 0.;
 *      x = SVDecomposition<double>::solve(A, B);   // solve Ax = b
 *      // x should be [2,2].
 *  \endcode
 *
 */
template <typename T>
class SVDecomposition
{
   public:

   /*!
    * Given A[m][n], b[m], solves A x = b using SVD.
    * returns solution vector x[n]
    * No input quantities are changed.
    */
   template<int M, int N>
   static FVector<T,N> solve( FMatrix<T,M,N> A, const FVector<T,M>& b );

   /*!
    * Given A[m][n], b[m], solves A x = b in the SVD form U W V'x = b
    * so x = V U'b/W
    * U[m][n], W[n][n] and V[n][n]
    * No input quantities are changed.
    */
   template<int M, int N>
   static FVector<T,N> svbksb( const FMatrix<T,M,N>& U, const FVector<T,N>& W, const FMatrix<T,N,N>& V,
                               const FVector<T,M>& b );

   /*!
    * Given matrix A[m][n], m>=n, using svd decomposition A = U W V' to get
    * U[m][n], W[n][n] and V[n][n], where U occupies the position of A.
    * NOTE: if m<n, A should be filled up to square with zero rows.
    *       A[m][n] has been destroyed by U[m][n] after the decomposition.
    */
   template<int M, int N>
   static void svdcmp( FMatrix<T,M,N>& A, FVector<T,N>& W, FMatrix<T,N,N>& V );

   protected:

   /*!
    * Computes (a^2 + b^2)^1/2 without destructive underflow or overflow
    *
    */
   static T pythag( const T a, const T b)
   {
      T absa, absb;

      absa = ::fabs(a);
      absb = ::fabs(b);

      if (absa> absb)
         return absa* ::sqrt(1.0 + pow2(absb/absa));
      else
         return (absb==0 ? 0.0 : absb* ::sqrt(T(1)+pow2(absa/absb)));
   }
};



template<typename T>
template<int M, int N>
FVector<T,N> SVDecomposition<T>::solve( FMatrix<T,M,N> A, const FVector<T,M>& b )
{
   FVector<T,N> W;
   FMatrix<T,N,N> V;
   svdcmp( A, W, V );
   return svbksb( A, W, V, b );
}


template <typename T>
template <int M, int N>
FVector<T,N> SVDecomposition<T>::svbksb( const FMatrix<T,M,N>& U, const FVector<T,N>& W, const FMatrix<T,N,N>& V,
                                         const FVector<T,M>& b )
{
   FVector<T,N> x, tmp;

   /* calculate tmp = (U^t b)/W */
   for (int j=1; j<=N; j++)
   {
      T s = 0;
      if (W(j)) /* Non-zero rsult only if Wj!=0 */
      {
         for (int i=1; i<=M; ++i)
            s += U(i,j)*b(i);
         s /= W(j); /* premultiplied by inverse W */
      }
      tmp(j)=s;
   }

   /* matrix multiply V tmp */
   // return dot( V, tmp );
   for (int j=1; j<=N; j++)
   {
      T s = 0;
      for (int i=1; i<=N; ++i)
         s += V(j,i)*tmp(i);
      x(j)=s;
   }

   return x;
}

template <typename T>
template <int M, int N>
void SVDecomposition<T>::svdcmp( FMatrix<T,M,N>& A, FVector<T,N>& W, FMatrix<T,N,N>& V )
{
   int flag, i, its, j, jj, k, l, nm;
   T anorm=0, c, f, g=0, h, s, scale=0, x, y, z;

   FVector<T,N> rv1;

   /* Householder reduction to bidigonal form */
   for (i=1; i<=N; ++i)
   {
      l = i+1;
      rv1(i) = scale*g;
      g = s = scale = 0;
      if (i <= M)
      {
         for (k=i; k<=M; k++)
            scale += ::fabs(A(k,i));
         if (scale)
         {
            for (k=i; k<=M; k++)
            {
               A(k,i) /= scale;
               s += pow2(A(k,i));
            }
            f = A(i,i);
            g = -SVD_SIGN(::sqrt(s),f);
            h = f*g-s;
            A(i,i) = f-g;
            for (j=l; j<=N; j++)
            {
               for (s=0.0, k=i; k<=M; k++)
                  s += A(k,i)*A(k,j);
               f=s/h;
               for (k=i; k<=M; k++)
                  A(k,j) += f*A(k,i);
            }
            for (k=i; k<=M; k++)
               A(k,i) *= scale;
         }
      }
      W(i) = scale * g;
      g = s = scale = 0;
      if (i <= M && i != N)
      {
         for (k=l; k<=N; k++)
            scale += ::fabs(A(i,k));
         if (scale)
         {
            for (k=l; k<=N; k++)
            {
               A(i,k) /= scale;
               s += pow2(A(i,k));
            }
            f = A(i,l);
            g = -SVD_SIGN(::sqrt(s),f);
            h = f*g - s;
            A(i,l) = f-g;
            for (k=l; k<=N; k++)
               rv1(k) = A(i,k)/h;
            for (j=l; j<=M; j++)
            {
               for (s=0.0, k=l; k<=N; k++)
                  s += A(j,k)*A(i,k);
               for (k=l; k<=N; k++)
                  A(j,k) += s*rv1(k);
            }
            for (k=l; k<=N; k++)
               A(i,k) *= scale;
         }
      }
      anorm = std::max(anorm, (::fabs(W(i))+::fabs(rv1(i))));
   }

   /* Accumulation of right-hand transformations */
   for (i=N; i>=1; i--)
   {
      if (i < N)
      {
         if (g)
         {
            /* double division to avoid possible underflow */
            for (j=l; j<=N; j++)
               V(j,i) = (A(i,j)/A(i,l))/g;
            for (j=l; j<=N; j++)
            {
               for (s=0.0, k=l; k<=N; k++)
                  s += A(i,k)*V(k,j);
               for (k=l; k<=N; k++)
                  V(k,j) += s*V(k,i);
            }
         }
         for (j=l; j<=N; j++)
            V(i,j) = V(j,i) = 0;
      }
      V(i,i) = T(1);
      g = rv1(i);
      l = i;
   }

   /* Accumulation of left-hand transformations */
   for (i=std::min(M, N); i>=1; i--)
   {
      l = i+1;
      g = W(i);
      for (j=l; j<=N; j++)
         A(i,j) = 0;
      if (g)
      {
         g = T(1)/g;
         for (j=l; j<=N; j++)
         {
            for (s=0.0, k=l; k<=M; k++)
               s += A(k,i)*A(k,j);
            f = (s/A(i,i))*g;
            for (k=i; k<=M; k++)
               A(k,j) += f*A(k,i);
         }
         for (j=i; j<=M; j++)
            A(j,i) *= g;
      }
      else
         for (j=i; j<=M; j++)
            A(j,i) = 0;
      A(i,i) += T(1);
   }

   /* diagonalization of the bidigonal form */

   /* loop over singlar values */
   for (k=N; k>=1; k--)
   {
      /* loop over allowed iterations */
      for (its=1; its<=30; its++)
      {
         flag = 1;
         /* test for splitting */
         for (l=k; l>=1; l--)
         {
            nm = l-1; /* note that rv1[l] is always zero */
            if ((T)(::fabs(rv1(l))+anorm) == anorm)
            {
               flag=0;
               break;
            }
            if ((T)(::fabs(W(nm))+anorm) == anorm)
            break;
         }
         if (flag)
         {
            /* cancellation of rv1[l], if l>1 */
            c=0.0;
            s=1.0;
            for (i=l; i<=k; i++)
            {
               f = s*rv1(i);
               rv1(i) = c*rv1(i);
               if ((T)(::fabs(f)+anorm) == anorm)
                  break;
               g = W(i);
               h = pythag(f, g);
               W(i) = h;
               h = T(1)/h;
               c = g*h;
               s = -f*h;
               for (j=1; j<=M; j++)
               {
                  y=A(j,nm);
                  z=A(j,i);
                  A(j,nm) = y*c+z*s;
                  A(j,i) = z*c-y*s;
               }
            }
         }
         z = W(k);
         if (l == k)
         {
            /* convergence */
            if (z < 0.0)
            {
               W(k) = -z;
               for (j=1; j<=N; j++)
                  V(j,k) = -V(j,k);
            }
            break;
         }
         if (its == 30)
         throw DivergenceException("SVD did not convergence in 30 iterations");
         x = W(l); /* shift from bottom 2-by-2 minor */
         nm = k-1;
         y = W(nm);
         g = rv1(nm);
         h = rv1(k);
         f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
         g = pythag(f, 1.0);
         /* next QR transformation */
         f = ((x-z)*(x+z)+h*((y/(f+SVD_SIGN(g,f)))-h))/x;
         c = s = T(1);
         for (j=l; j<=nm; j++)
         {
            i = j+1;
            g = rv1(i);
            y = W(i);
            h = s*g;
            g = c*g;
            z = pythag(f, h);
            rv1(j) = z;
            c = f/z;
            s = h/z;
            f = x*c+g*s;
            g = g*c-x*s;
            h = y*s;
            y *= c;
            for (jj=1; jj<=N; jj++)
            {
               x = V(jj,j);
               z = V(jj,i);
               V(jj,j) = x*c+z*s;
               V(jj,i) = z*c-x*s;
            }
            z = pythag(f, h);
            W(j)=z; /* rotation can be arbitrary id z=0 */
            if (z)
            {
               z = 1.0/z;
               c = f*z;
               s = h*z;
            }
            f = c*g+s*y;
            x = c*y-s*g;
            for (jj=1; jj<=M; jj++)
            {
               y=A(jj,j);
               z=A(jj,i);
               A(jj,j) = y*c+z*s;
               A(jj,i) = z*c-y*s;
            }
         }
         rv1(l) = 0;
         rv1(k) = f;
         W(k) = x;
      }
   }
}
//@}

#undef SVD_SIGN

}

#endif  // __LTL_FSVD__
