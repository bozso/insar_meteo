/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: blas.h 565 2015-06-11 18:01:22Z drory $
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


// ====================================================================

#ifndef __LTL_BLAS_H__
#define __LTL_BLAS_H__

#if !defined(HAVE_BLAS)
#  error "LTL was configured without BLAS support. Rerun configure, and if auto-detection of BLAS library fails, use --with-blas[=LIB] option."
#endif

#include <ltl/acconfig.h>
#ifdef HAVE_COMPLEX
#  include <complex>
#endif

/// \cond DOXYGEN_IGNORE
extern "C"
{
   /*
    *  We provide our own BLAS prototypes, since finding the right header file
    *  on all supported systems and BLAS versions is a major pain in the ass.
    */
   double sdot_(const int *n, const float *x, const int *incx, const float *y,
                const int *incy);

   double ddot_(const int *n, const double *x, const int *incx, const double *y,
                const int *incy);

   void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
               float *y, const int *incy);

   void daxpy_(const int *n, const double *alpha, const double *x,
               const int *incx, double *y, const int *incy);

   void dgemv_(const char *transA, const int *m, const int *n,
               const double *alpha, const double *A, const int *ldA,
               const double *x, const int *incx, const double *beta, double *y,
               const int *incy);

   void sgemv_(const char *transA, const int *m, const int *n, const float *alpha,
               const float *A, const int *ldA, const float *x, const int *incx,
               const float *beta, float *y, const int *incy);

   void dsymv_(const char *uplo, const int *n, const double *alpha, const double *A,
               const int *ldA, const double *x, const int *incx, const double *beta,
               double *y, const int *incy);

   void ssymv_(const char *uplo, const int *n, const float *alpha, const float *A,
               const int *ldA, const float *x, const int *incx, const float *beta,
               float *y, const int *incy);

   void dgbmv_(const char *transA, const int *M, const int *N, const int *kl, const int *ku,
               const double *alpha, const double *A, const int *LDA, const double *x, const int *xincr,
               const double *beta, double *y, const int *yincr);

   void sgbmv_(const char *transA, const int *M, const int *N, const int *kl, const int *ku,
               const float *alpha, const float *A, const int *LDA,
               const float *x, const int *xincr, const float *beta,
               float *y, const int *yincr);

   void dgemm_(const char *transA, const char *transB, const int *m, const int *n,
               const int *k, const double *alpha, const double *A, const int *ldA,
               const double *B, const int *ldB, const double *beta, double *C,
               const int *ldC);

   void sgemm_(const char *transA, const char *transB, const int *m, const int *n,
               const int *k, const float *alpha, const float *A, const int *ldA,
               const float *B, const int *ldB, const float *beta, float *C,
               const int *ldC);

#ifdef HAVE_COMPLEX
#  define CFLOAT std::complex<float>
#  define CDOUBLE std::complex<double>

   CFLOAT cdot_(const int *n, const CFLOAT *x, const int *incx, const CFLOAT *y,
                const int *incy);

   CDOUBLE zdot_(const int *n, const CDOUBLE *x, const int *incx, const CDOUBLE *y,
                 const int *incy);

   void caxpy_(const int *n, const CFLOAT *alpha, const CFLOAT *x, const int *incx,
               CFLOAT *y, const int *incy);

   void zaxpy_(const int *n, const CDOUBLE *alpha, const CDOUBLE *x,
               const int *incx, CDOUBLE *y, const int *incy);

   void zgemv_(const char *transA, const int *m, const int *n,
               const CDOUBLE *alpha, const CDOUBLE *A, const int *ldA,
               const CDOUBLE *x, const int *incx, const CDOUBLE *beta, CDOUBLE *y,
               const int *incy);

   void cgemv_(const char *transA, const int *m, const int *n, const CFLOAT *alpha,
               const CFLOAT *A, const int *ldA, const CFLOAT *x, const int *incx,
               const CFLOAT *beta, CFLOAT *y, const int *incy);

   void zsymv_(const char *uplo, const int *n, const CDOUBLE *alpha, const CDOUBLE *A,
               const int *ldA, const CDOUBLE *x, const int *incx, const CDOUBLE *beta,
               CDOUBLE *y, const int *incy);

   void csymv_(const char *uplo, const int *n, const CFLOAT *alpha, const CFLOAT *A,
               const int *ldA, const CFLOAT *x, const int *incx, const CFLOAT *beta,
               CFLOAT *y, const int *incy);

   void zgbmv_(const char *transA, const int *M, const int *N, const int *kl, const int *ku,
               const CDOUBLE *alpha, const CDOUBLE *A, const int *LDA, const CDOUBLE *x, const int *xincr,
               const CDOUBLE *beta, double *y, const int *yincr);

   void cgbmv_(const char *transA, const int *M, const int *N, const int *kl, const int *ku,
               const CFLOAT *alpha, const CFLOAT *A, const int *LDA,
               const CFLOAT *x, const int *xincr, const CFLOAT *beta,
               CFLOAT *y, const int *yincr);

   void zgemm_(const char *transA, const char *transB, const int *m, const int *n,
               const int *k, const CDOUBLE *alpha, const CDOUBLE *A, const int *ldA,
               const CDOUBLE *B, const int *ldB, const CDOUBLE *beta, CDOUBLE *C,
               const int *ldC);

   void cgemm_(const char *transA, const char *transB, const int *m, const int *n,
               const int *k, const CFLOAT *alpha, const CFLOAT *A, const int *ldA,
               const CFLOAT *B, const int *ldB, const CFLOAT *beta, CFLOAT *C,
               const int *ldC);
#endif
}

namespace ltl {
/// \endcond

//@{

/*!
 * \ingroup blas
 * \addindex BLAS level 1 calls
 *
 *  BLAS Level 1 functions
 */

/// \cond DOXYGEN_IGNORE
// dispatchers for BLAS {s,d,c,z}dot_ for different data types
template <typename T>
struct blas_dot_dispatch
{
   static inline T call( const int *n, const T *x, const int *incx, const T *y,
         const int *incy )
   { LTL_ASSERT_(false, "Invalid type in blas_dot_dispatch"); }
};
// DOUBLE
template <> struct blas_dot_dispatch<double>
{
   typedef double value_type;
   static inline double call(const int *n, const double *x, const int *incx,
                             const double *y, const int *incy)
   {
      return ddot_(n, x, incx, y, incy);
   }
};
// FLOAT
template <> struct blas_dot_dispatch<float>
{
   typedef double value_type;
   static inline float call(const int *n, const float *x, const int *incx,
                            const float *y, const int *incy)
   {
      return sdot_(n, x, incx, y, incy);
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_dot_dispatch<CDOUBLE>
{
   typedef CDOUBLE value_type;
   static inline CDOUBLE call(const int *n, const CDOUBLE *x, const int *incx,
                              const CDOUBLE *y, const int *incy)
   {
      return zdot_(n, x, incx, y, incy);
   }
};
// COMPLEX FLOAT
template <> struct blas_dot_dispatch<CFLOAT>
{
   typedef CDOUBLE value_type;
   static inline CDOUBLE call(const int *n, const CFLOAT *x, const int *incx,
                              const CFLOAT *y, const int *incy)
   {
      return cdot_(n, x, incx, y, incy);
   }
};
#endif
/// \endcond

/*!
 *  Compute the dot product of two n-element vectors x and y:
 *  x dot y = SUM(i=1...n,x(i)y(i)) = x(1)y(1) + x(2)y(2) + ... + x(n)y(n)
 *
 *  The \c MArrays x and y may have strides unequal 1, hence they may be
 *  views, slices, or subarrays of higher-dimensional \c MArrays
 *
 *  The order of operations is different from the order in a sequential evalua-
 *  tion of the dot product. The final result can differ from the result of a
 *  sequential evaluation. Returns the value in double-precision.
 */
template <typename T> typename blas_dot_dispatch<T>::value_type
blas_dot( const MArray<T,1>& x, const MArray<T,1>& y )
{
   LTL_ASSERT(x.nelements() == y.nelements(), "Length of Arrays not equal.");

   const int nx = x.nelements();
   const int xincr = x.stride(1);
   const int yincr = y.stride(1);

   return blas_dot_dispatch<T>::call( &nx, x.data(), &xincr, y.data(), &yincr);
}



/// \cond DOXYGEN_IGNORE
// dispatchers for BLAS {s,d,c,z}axpy_ for different data types
template <typename T>
struct blas_axpy_dispatch
{
   static inline void call( const int *n, const T *alpha, const T *x, const int *incx,
                            T *y, const int *incy )
   { LTL_ASSERT_(false, "Invalid type in blas_axpy_dispatch"); }
};
// DOUBLE
template <> struct blas_axpy_dispatch<double>
{
   static inline void call( const int *n, const double *alpha, const double *x, const int *incx,
                            double *y, const int *incy )
   {
      daxpy_( n, alpha, x, incx, y, incy );
   }
};
// FLOAT
template <> struct blas_axpy_dispatch<float>
{
   static inline void call( const int *n, const float *alpha, const float *x, const int *incx,
                            float *y, const int *incy )
   {
      saxpy_( n, alpha, x, incx, y, incy );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_axpy_dispatch<CDOUBLE>
{
   static inline void call( const int *n, const CDOUBLE *alpha, const CDOUBLE *x, const int *incx,
                            CDOUBLE *y, const int *incy )
   {
      zaxpy_( n, alpha, x, incx, y, incy );
   }
};
// COMPLEX FLOAT
template <> struct blas_axpy_dispatch<CFLOAT>
{
   static inline void call( const int *n, const CFLOAT *alpha, const CFLOAT *x, const int *incx,
                            CFLOAT *y, const int *incy )
   {
      caxpy_( n, alpha, x, incx, y, incy );
   }
};
#endif
/// \endcond

/*!
 *  The _AXPY functions compute the following scalar-vector product and sum:
 *  y = alpha*x+y
 *  where alpha is a scalar, and x and y are single-precision vectors.
 *
 *  If any element of x or the scalar alpha share a memory location with an
 *  element of y, the results are unpredictable.
 *
 *  The \c MArrays x and y may have strides unequal 1, hence they may be
 *  views, slices, or subarrays of higher-dimensional \c MArrays
 *
 *  The vector y is overwritted with the result of the calculation.
 */
template <typename T>
void blas_axpy(const MArray<T,1>& x, const T alpha, MArray<T,1>& y)
{
   LTL_ASSERT(x.nelements() == y.nelements(), "Length of Arrays not equal.");

   const int nx = x.nelements();
   const int xincr = x.stride(1);
   const int yincr = y.stride(1);

   blas_axpy_dispatch<T>::call( &nx, &alpha, x.data(), &xincr, y.data(), &yincr );
}

//@}

//@{

  /*!
   * \ingroup blas
   * \addindex BLAS level 2 calls
   *
   *  BLAS Level 2 functions
   */

/// \cond DOXYGEN_IGNORE
// dispatchers for BLAS {s,d,c,z}gemv_ for different data types
template <typename T>
struct blas_gemv_dispatch
{
   static inline void call( const char *transA, const int *m, const int *n, const T *alpha,
                            const T *A, const int *ldA, const T *x, const int *incx,
                            const T *beta, T *y, const int *incy )
   { LTL_ASSERT_(false, "Invalid type in blas_gemv_dispatch"); }
};
// DOUBLE
template <> struct blas_gemv_dispatch<double>
{
   static inline void call( const char *transA, const int *m, const int *n, const double *alpha,
                            const double *A, const int *ldA, const double *x, const int *incx,
                            const double *beta, double *y, const int *incy )
   {
      dgemv_( transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
// FLOAT
template <> struct blas_gemv_dispatch<float>
{
   static inline void call( const char *transA, const int *m, const int *n, const float *alpha,
                            const float *A, const int *ldA, const float *x, const int *incx,
                            const float *beta, float *y, const int *incy )
   {
      sgemv_( transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_gemv_dispatch<CDOUBLE>
{
   static inline void call( const char *transA, const int *m, const int *n, const CDOUBLE *alpha,
                            const CDOUBLE *A, const int *ldA, const CDOUBLE *x, const int *incx,
                            const CDOUBLE *beta, CDOUBLE *y, const int *incy )
   {
      zgemv_( transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
// COMPLEX FLOAT
template <> struct blas_gemv_dispatch<CFLOAT>
{
   static inline void call( const char *transA, const int *m, const int *n, const CFLOAT *alpha,
                            const CFLOAT *A, const int *ldA, const CFLOAT *x, const int *incx,
                            const CFLOAT *beta, CFLOAT *y, const int *incy )
   {
      cgemv_( transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
#endif
// dispatchers for BLAS {s,d,c,z}symv_ for different data types
template <typename T>
struct blas_symv_dispatch
{
   static inline void call( const char *uplo, const int *n, const T *alpha,
                            const T *A, const int *ldA, const T *x, const int *incx,
                            const T *beta, T *y, const int *incy )
   { LTL_ASSERT_(false, "Invalid type in blas_symv_dispatch"); }
};
// DOUBLE
template <> struct blas_symv_dispatch<double>
{
   static inline void call( const char *uplo, const int *n, const double *alpha,
                            const double *A, const int *ldA, const double *x, const int *incx,
                            const double *beta, double *y, const int *incy )
   {
      dsymv_( uplo, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
// FLOAT
template <> struct blas_symv_dispatch<float>
{
   static inline void call( const char *uplo, const int *n, const float *alpha,
                            const float *A, const int *ldA, const float *x, const int *incx,
                            const float *beta, float *y, const int *incy )
   {
      ssymv_( uplo, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_symv_dispatch<CDOUBLE>
{
   static inline void call( const char *uplo, const int *n, const CDOUBLE *alpha,
                            const CDOUBLE *A, const int *ldA, const CDOUBLE *x, const int *incx,
                            const CDOUBLE *beta, CDOUBLE *y, const int *incy )
   {
      zsymv_( uplo, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
// COMPLEX FLOAT
template <> struct blas_symv_dispatch<CFLOAT>
{
   static inline void call( const char *uplo, const int *n, const CFLOAT *alpha,
                            const CFLOAT *A, const int *ldA, const CFLOAT *x, const int *incx,
                            const CFLOAT *beta, CFLOAT *y, const int *incy )
   {
      csymv_( uplo, n, alpha, A, ldA, x, incx, beta, y, incy );
   }
};
#endif
/// \endcond

// dispatchers for BLAS {s,d,c,z}symv_ for different data types
template <typename T>
struct blas_gbmv_dispatch
{
   static inline void call( const char *transA, const int *M, const int *N, const int *kl, const int *ku,
                            const T *alpha, const T *A, const int *LDA, const T *x, const int *xincr,
                            const T *beta, T *y, const int *yincr )
   { LTL_ASSERT_(false, "Invalid type in blas_gbmv_dispatch"); }
};
// DOUBLE
template <> struct blas_gbmv_dispatch<double>
{
   static inline void call( const char *transA, const int *M, const int *N, const int *kl, const int *ku,
                            const double *alpha, const double *A, const int *LDA, const double *x, const int *xincr,
                            const double *beta, double *y, const int *yincr )
   {
      dgbmv_(transA, M, N, kl, ku, alpha, A, LDA, x, xincr, beta, y, yincr);
   }
};
// FLOAT
template <> struct blas_gbmv_dispatch<float>
{
   static inline void call( const char *transA, const int *M, const int *N, const int *kl, const int *ku,
                            const float *alpha, const float *A, const int *LDA,
                            const float *x, const int *xincr, const float *beta,
                            float *y, const int *yincr )
   {
      sgbmv_(transA, M, N, kl, ku, alpha, A, LDA, x, xincr, beta, y, yincr);
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_gbmv_dispatch<CDOUBLE>
{
   static inline void call( const char *transA, const int *M, const int *N, const int *kl, const int *ku,
                            const CDOUBLE *alpha, const CDOUBLE *A, const int *LDA, const CDOUBLE *x, const int *xincr,
                            const CDOUBLE *beta, double *y, const int *yincr )
   {
      zgbmv_(transA, M, N, kl, ku, alpha, A, LDA, x, xincr, beta, y, yincr);
   }
};
// COMPLEX FLOAT
template <> struct blas_gbmv_dispatch<CFLOAT>
{
   static inline void call( const char *transA, const int *M, const int *N, const int *kl, const int *ku,
                            const CFLOAT *alpha, const CFLOAT *A, const int *LDA,
                            const CFLOAT *x, const int *xincr, const CFLOAT *beta,
                            CFLOAT *y, const int *yincr )
   {
      cgbmv_(transA, M, N, kl, ku, alpha, A, LDA, x, xincr, beta, y, yincr);
   }
};
#endif
/// \endcond

/*!
 *  The \c Xgemv() routine compute a matrix-vector product for of
 *  either a general matrix or its transpose:
 *  y  =  alpha*Ax + beta*y
 *  y  =  alpha*transp(A)*x + beta*y
 *
 *  The \c MArrays x and y may have strides unequal 1, hence they may be
 *  views, slices, or subarrays of higher-dimensional \c MArrays.
 *  The matrix \c A needs to have contiguous storage.
 *
 *  The vector y is overwritted with the result of the calculation.
 *  If transA==false, the length of y has to be M, otherwise N.
 */
template <typename T>
void blas_gemv( const T alpha, const MArray<T,2>& A, const MArray<T,1>& x,
                const T beta, MArray<T,1>& y, const bool transA=false )
{
   LTL_ASSERT(A.length(transA ? 1 : 2) == x.length(1),
              "Matrix rows != x Vector length.");
   LTL_ASSERT(A.length(transA ? 2 : 1) == y.length(1),
              "Matrix columns != y vector length.");
   LTL_ASSERT( A.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const int N = A.length(2);
   const int M = A.length(1);
   const int xincr = x.stride(1);
   const int yincr = y.stride(1);
   const char tA = (transA ? 'T' : 'N');

   blas_gemv_dispatch<T>::call( &tA, &M, &N, &alpha, A.data(), &M, x.data(), &xincr, &beta, y.data(), &yincr );
}

/*!
 *  The \c Xgemv() routine compute a matrix-vector product for double-precision data of
 *  either a general matrix or its transpose:
 *  y  =  Ax
 *  y  =  transp(A)*x
 *
 *  The \c MArrays x may have strides unequal 1, hence it may be a
 *  view, slice, or subarray of a higher-dimensional \c MArray
 *  The matrix \c A needs to have contiguous storage.
 *
 *  The result of the calculation is returned in a newly allocated \c MArray.
 */
template <typename T>
MArray<T,1> blas_gemv( const MArray<T,2>& A, const MArray<T,1>& x, const bool transA=false )
{
   LTL_ASSERT( A.length(transA ? 1 : 2) == x.length(1), "Matrix rows != x Vector length.");
   LTL_ASSERT( A.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const T alpha = 1.0;
   const T beta = 0.0;
   const int N = A.length(2);
   const int M = A.length(1);
   const int xincr = x.stride(1);
   const int yincr = 1;
   const char tA = (transA ? 'T' : 'N');

   MArray<T,1> y(transA ? A.length(2) : A.length(1) );
   blas_gemv_dispatch<T>::call( &tA, &M, &N, &alpha, A.data(), &M, x.data(), &xincr, &beta, y.data(), &yincr);

   return y;
}

//@}

/*!
 *  The \c Xsymv() routine compute a matrix-vector product for of
 *  a symmetric or hermitian matrix:
 *  y  =  alpha*Ax + beta*y
 *
 *  The \c MArrays x and y may have strides unequal 1, hence they may be
 *  views, slices, or subarrays of higher-dimensional \c MArrays.
 *  The matrix \c A needs to have contiguous storage.
 *
 *  The vector y is overwritted with the result of the calculation.
 */
template <typename T>
void blas_symv( const T alpha, const MArray<T,2>& A, const MArray<T,1>& x,
                const T beta, MArray<T,2>& y )
{
   LTL_ASSERT(A.length(1) == x.length(1),
              "Matrix rows != x Vector length.");
   LTL_ASSERT(A.length(2) == y.length(1),
              "Matrix columns != y vector length.");
   LTL_ASSERT( A.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const int N = A.length(1);
   const int xincr = x.stride(1);
   const int yincr = y.stride(1);
   const char uplo = 'U';

   blas_symv_dispatch<T>::call( &uplo, &N, &alpha, A.data(), &N, x.data(), &xincr, &beta, y.data(), &yincr );
}

/*!
 *  The \c Xsymv() routine compute a matrix-vector product for double-precision data of
 *  either a general matrix or its transpose:
 *  y  =  Ax
 *
 *  The \c MArrays x may have strides unequal 1, hence it may be a
 *  view, slice, or subarray of a higher-dimensional \c MArray
 *  The matrix \c A needs to have contiguous storage.
 *
 *  The result of the calculation is returned in a newly allocated \c MArray.
 */
template <typename T>
MArray<T,1> blas_symv( const MArray<T,2>& A, const MArray<T,1>& x )
{
   LTL_ASSERT( A.length(1) == x.length(1), "Matrix rows != x Vector length.");
   LTL_ASSERT( A.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const T alpha = 1.0;
   const T beta = 0.0;
   const int N = A.length(1);
   const int xincr = x.stride(1);
   const int yincr = 1;
   const char uplo = 'U';

   MArray<T,1> y( A.length(1) );
   blas_symv_dispatch<T>::call( &uplo, &N, &alpha, A.data(), &N, x.data(), &xincr, &beta, y.data(), &yincr);

   return y;
}

//@}

/*!
 *  The \c Xgbmv() routine compute a matrix-vector product for single or double-precision data of
 *  either a general banded matrix or its transpose:
 *  y  =  Ax
 *
 *  The \c MArrays x may have strides unequal 1, hence it may be a
 *  view, slice, or subarray of a higher-dimensional \c MArray
 *  The matrix \c A needs to have contiguous storage.
 *
 *  The result of the calculation is returned in a newly allocated \c MArray.
 */
template <typename T>
MArray<T,1> blas_gbmv( const MArray<T,2>& A, const MArray<T,1>& x, const int M, const int ku, const int kl, const bool transA )
{
   LTL_ASSERT( A.length(2) == x.length(1), "Matrix columns != x Vector length.");
   LTL_ASSERT( A.length(1) == kl+ku+1, "Bandwidths inconsistent with compact storage size.");
   LTL_ASSERT( A.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const T alpha = 1.0;
   const T beta = 0.0;
   const int N = A.length(2);
   const int LDA = kl+ku+1;
   const int xincr = x.stride(1);
   const int yincr = 1;
   const char tA = (transA ? 'T' : 'N');

   MArray<T,1> y( M );
   blas_gbmv_dispatch<T>::call( &tA, &M, &N, &kl, &ku, &alpha, A.data(), &LDA, x.data(), &xincr, &beta, y.data(), &yincr);

   return y;
}

//@}


//@{

  /*!
   * \ingroup blas
   * \addindex BLAS level 1 calls
   *
   *  BLAS Level 3 functions
   */

/// \cond DOXYGEN_IGNORE
// dispatchers for BLAS {s,d,c,z}gemv_ for different data types
template <typename T>
struct blas_gemm_dispatch
{
   static inline void call( const char *transA, const char *transB, const int *m, const int *n,
                            const int *k, const T *alpha, const T *A, const int *ldA,
                            const T *B, const int *ldB, const T *beta, T *C,
                            const int *ldC )
   { LTL_ASSERT_(false, "Invalid type in blas_gemm_dispatch"); }
};
// DOUBLE
template <> struct blas_gemm_dispatch<double>
{
   static inline void call( const char *transA, const char *transB, const int *m, const int *n,
                            const int *k, const double *alpha, const double *A, const int *ldA,
                            const double *B, const int *ldB, const double *beta, double *C,
                            const int *ldC )
   {
      dgemm_( transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
   }
};
// FLOAT
template <> struct blas_gemm_dispatch<float>
{
   static inline void call( const char *transA, const char *transB, const int *m, const int *n,
                            const int *k, const float *alpha, const float *A, const int *ldA,
                            const float *B, const int *ldB, const float *beta, float *C,
                            const int *ldC )
   {
      sgemm_( transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX DOUBLE
template <> struct blas_gemm_dispatch<CDOUBLE>
{
   static inline void call( const char *transA, const char *transB, const int *m, const int *n,
                            const int *k, const CDOUBLE *alpha, const CDOUBLE *A, const int *ldA,
                            const CDOUBLE *B, const int *ldB, const CDOUBLE *beta, CDOUBLE *C,
                            const int *ldC )
   {
      zgemm_( transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
   }
};
// COMPLEX FLOAT
template <> struct blas_gemm_dispatch<CFLOAT>
{
   static inline void call( const char *transA, const char *transB, const int *m, const int *n,
                            const int *k, const CFLOAT *alpha, const CFLOAT *A, const int *ldA,
                            const CFLOAT *B, const int *ldB, const CFLOAT *beta, CFLOAT *C,
                            const int *ldC )
   {
      cgemm_( transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
   }
};
#endif
/// \endcond

/*!
 *  This routine performs the following operation:
 *  C  = alpha(op)A(op)B + beta*C
 *  where (op)(X) = X, transp(X), alpha and beta are scalars,
 *  and A, B, and C are matrices.
 *
 *  (op)A is an m by k matrix, (op)B is a k by n matrix, and C is an m by n matrix.
 *  The matrices need to have contiguous storage.
 *
 *  A and B are unchanged on exit, C is overwritten with A dot B.
 *
 */
template <typename T> void blas_gemm( const T alpha, const MArray<T,2>& A, const MArray<T,2>& B, const T beta,
                                      MArray<T,2>& C, const bool transA=false, const bool transB=false )
{
   LTL_ASSERT(A.length(transA ? 1 : 2) == B.length(transB ? 2 : 1),
              "Matrix A and B are not compatible.");
   LTL_ASSERT(A.length(transA ? 2 : 1) == C.length(1),
              "Matrix A and C are not compatible.");
   LTL_ASSERT(B.length(transB ? 1 : 2) == C.length(2),
              "Matrix B and C are not compatible.");
   LTL_ASSERT( A.isStorageContiguous() && B.isStorageContiguous() && C.isStorageContiguous(), "Matrix has non-contiguous storage." );

   const int M = A.length(transA?2:1);
   const int N = B.length(transB ? 1 : 2);
   const int K = A.length(transA ? 1 : 2);
   const char tA = (transA ? 'T' : 'N');
   const char tB = (transB ? 'T' : 'N');

   blas_gemm_dispatch<T>::call( &tA, &tB, &M, &N, &K, &alpha, A.data(), (transA ? &K : &M),
                                B.data(), (transB ? &N : &K), &beta, C.data(), &M);
}

//@}

#ifdef HAVE_COMPLEX
# undef CFLOAT
# undef CDOUBLE
#endif

}


#endif // #ifndef __LTL_BLAS_H__
