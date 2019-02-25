/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: lapack.h 565 2015-06-11 18:01:22Z drory $
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

#ifndef __LTL_LAPACK_H__
#define __LTL_LAPACK_H__

#if !defined(HAVE_LAPACK)
#  error "LTL was configured without LAPACK support. Rerun configure, and if auto-detection of LAPACK library fails, use --with-lapack[=LIB] option."
#endif

#include <ltl/acconfig.h>
#ifdef HAVE_COMPLEX
#  include <complex>
#endif

/// \cond DOXYGEN_IGNORE
extern "C"
{
   /*
    *  We provide our own LAPACK prototypes, since finding the right header file
    *  on all supported systems and LAPACK versions is a major pain in the ass.
    */
   void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);
   void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);

   void sgetrf_( int *m, int *n, float *a, int *lda, int *ipiv, int *info );
   void dgetrf_( int *m, int *n, double *a, int *lda, int *ipiv, int *info );

   void sgetri_( int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info );
   void dgetri_( int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info );

   void ssyev_(char *job, char *uplo, int *n, float *a, int *lda, float *w,
	       float *work, int *lwork, int *info);
   void dsyev_(char *job, char *uplo, int *n, double *a, int *lda, double *w,
	       double *work, int *lwork, int *info);

   void ssbev_(char *job, char *uplo, int *n, int *kd, float *ab, int *ldab, float *w,
	       float *z, int *ldz, float *work, int *info);
   void dsbev_(char *job, char *uplo, int *n, int *kd, double *ab, int *ldab, double *w,
	       double *z, int *ldz, double *work, int *info);

#ifdef HAVE_COMPLEX
#  define CFLOAT std::complex<float>
#  define CDOUBLE std::complex<double>

   void cgesv_(int *n, int *nrhs, CFLOAT *a, int *lda, int *ipiv, CFLOAT *b, int *ldb, int *info);
   void zgesv_(int *n, int *nrhs, CDOUBLE *a, int *lda, int *ipiv, CDOUBLE *b, int *ldb, int *info);

   void cgetrf_( int *m, int *n, CFLOAT *a, int *lda, int *ipiv, int *info );
   void zgetrf_( int *m, int *n, CDOUBLE *a, int *lda, int *ipiv, int *info );

   void cgetri_( int *n, CFLOAT *a, int *lda, int *ipiv, CFLOAT *work, int *lwork, int *info );
   void zgetri_( int *n, CDOUBLE *a, int *lda, int *ipiv, CDOUBLE *work, int *lwork, int *info );

#endif
}

namespace ltl {
/// \endcond

//@{

/*!
 * \ingroup blas
 * \addindex LAPACK
 *
 *  LAPACK Functions
 */

/// \cond DOXYGEN_IGNORE
// dispatchers for LAPACK {s,d,c,z}gesv_ for different data types
template <typename T>
struct lapack_gesv_dispatch
{
   static inline void call( int *n, int *nrhs, T *a, int *lda, int *ipiv, T *b,
                            int *ldb, int *info )
   { LTL_ASSERT_(false, "Invalid type in lapack_gesv_dispatch"); }
};
// DOUBLE
template <> struct lapack_gesv_dispatch<double>
{
   static inline void call( int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b,
                            int *ldb, int *info)
   {
      return dgesv_( n, nrhs, a, lda, ipiv, b, ldb, info );
   }
};
// FLOAT
template <> struct lapack_gesv_dispatch<float>
{
   static inline void call( int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b,
                              int *ldb, int *info )
   {
      return sgesv_( n, nrhs, a, lda, ipiv, b, ldb, info );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX FLOAT
template <> struct lapack_gesv_dispatch<CFLOAT>
{
   static inline void call( int *n, int *nrhs, CFLOAT *a, int *lda, int *ipiv, CFLOAT *b,
                              int *ldb, int *info )
   {
      return cgesv_( n, nrhs, a, lda, ipiv, b, ldb, info );
   }
};
// COMPLEX DOUBLE
template <> struct lapack_gesv_dispatch<CDOUBLE>
{
   static inline void call( int *n, int *nrhs, CDOUBLE *a, int *lda, int *ipiv, CDOUBLE *b,
                              int *ldb, int *info )
   {
      return zgesv_( n, nrhs, a, lda, ipiv, b, ldb, info );
   }
};
#endif
/// \endcond

/// \cond DOXYGEN_IGNORE
// dispatchers for LAPACK {s,d,c,z}getri_ for different data types
template <typename T>
struct lapack_getrf_dispatch
{
   static inline void call( int *m, int *n, int, T *a, int *lda, int *ipiv, int *info )
   { LTL_ASSERT_(false, "Invalid type in lapack_getri_dispatch"); }
};
// DOUBLE
template <> struct lapack_getrf_dispatch<double>
{
   static inline void call( int *m, int *n, double *a, int *lda, int *ipiv, int *info )
   {
      return dgetrf_( m, n, a, lda, ipiv, info );
   }
};
// FLOAT
template <> struct lapack_getrf_dispatch<float>
{
   static inline void call( int *m, int *n, float *a, int *lda, int *ipiv, int *info )
   {
      return sgetrf_( m, n, a, lda, ipiv, info );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX FLOAT
template <> struct lapack_getrf_dispatch<CFLOAT>
{
   static inline void call( int *m, int *n, CFLOAT *a, int *lda, int *ipiv, int *info )
   {
      return cgetrf_( m, n, a, lda, ipiv, info );
   }
};
// COMPLEX DOUBLE
template <> struct lapack_getrf_dispatch<CDOUBLE>
{
   static inline void call( int *m, int *n, CDOUBLE *a, int *lda, int *ipiv, int *info )
   {
      return zgetrf_( m, n, a, lda, ipiv, info );
   }
};
#endif
/// \endcond

/// \cond DOXYGEN_IGNORE
// dispatchers for LAPACK {s,d,c,z}getri_ for different data types
template <typename T>
struct lapack_getri_dispatch
{
   static inline void call( int *n, int, T *a, int *lda, int *ipiv, T *work, T *lwork, int *info )
   { LTL_ASSERT_(false, "Invalid type in lapack_getri_dispatch"); }
};
// DOUBLE
template <> struct lapack_getri_dispatch<double>
{
   static inline void call( int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info )
   {
      return dgetri_( n, a, lda, ipiv, work, lwork, info );
   }
};
// FLOAT
template <> struct lapack_getri_dispatch<float>
{
   static inline void call( int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info )
   {
      return sgetri_( n, a, lda, ipiv, work, lwork, info );
   }
};
#ifdef HAVE_COMPLEX
// COMPLEX FLOAT
template <> struct lapack_getri_dispatch<CFLOAT>
{
   static inline void call( int *n, CFLOAT *a, int *lda, int *ipiv, CFLOAT *work, int *lwork, int *info )
   {
      return cgetri_( n, a, lda, ipiv, work, lwork, info );
   }
};
// COMPLEX DOUBLE
template <> struct lapack_getri_dispatch<CDOUBLE>
{
   static inline void call( int *n, CDOUBLE *a, int *lda, int *ipiv, CDOUBLE *work, int *lwork, int *info )
   {
      return zgetri_( n, a, lda, ipiv, work, lwork, info );
   }
};
#endif
/// \endcond

template <typename T>
struct lapack_syev_dispatch
{
   static inline void call( char *jobz, char *uplo, int *N, T *vec,
			    int &lda, T *val, T *work, int &lwork, int &info )
  { LTL_ASSERT_(false, "Invalid type in lapack_gesv_dispatch"); }
};
// DOUBLE
template <> struct lapack_syev_dispatch<double>
{
   static inline void call( char *jobz, char *uplo, int *N, double *vec,
			    int *lda, double *val, double *work, int *lwork, int *info )
  {
    return dsyev_( jobz, uplo, N, vec, lda, val, work, lwork, info );
  }
};
// FLOAT
template <> struct lapack_syev_dispatch<float>
{
   static inline void call( char *jobz, char *uplo, int *N, float *vec,
			    int *lda, float *val, float *work, int *lwork, int *info )
  {
     return ssyev_( jobz, uplo, N, vec, lda, val, work, lwork, info );
  }
};

template <typename T>
struct lapack_sbev_dispatch
{
  static inline void call( char *jobz, char *uplo, int *N, int *kd, T *AB, int &ldab,
			   T *val, T *vec, int *ldv, T *work, int &info )
  { LTL_ASSERT_(false, "Invalid type in lapack_gesv_dispatch"); }
};
// DOUBLE
template <> struct lapack_sbev_dispatch<double>
{
   static inline void call( char *jobz, char *uplo, int *N, int *kd, double *AB, int *ldab,
			    double *val, double *vec, int *ldv, double *work, int *info )
  {
     return dsbev_( jobz, uplo, N, kd, AB, ldab, val, vec, ldv, work, info );
  }
};
// FLOAT
template <> struct lapack_sbev_dispatch<float>
{
  static inline void call( char *jobz, char *uplo, int *N, int *kd, float *AB, int *ldab,
			   float *val, float *vec, int *ldv, float *work, int *info )
  {
     return ssbev_( jobz, uplo, N, kd, AB, ldab, val, vec, ldv, work, info );
  }
};

/*!
 *  GESV computes the solution to a system of linear equations
 *     A * x = b,
 *  where A is an N-by-N matrix and x and b are N-vectors.
 *
 *  LU decomposition with partial pivoting and row interchanges is
 *  used to factor A as
 *     A = P * L * U,
 *  where P is a permutation matrix, L is unit lower triangular, and U is
 *  upper triangular.  The factored form of A is then used to solve the
 *  system of equations A * x = b.
 *
 *  The matrix A is replaced by L and U (or A^-1, see below).
 *  The vector b is overwritten with the solution of Ax=b.
 *
 *  The permutation matrix is returned in \c ipiv. Call getri() to
 *  compute the inverse of \c A.
 */
template <typename T>
bool lapack_gesv( MArray<T,2>& A, MArray<T,1>& b, MArray<int,1>& ipiv )
{
   int N = A.length(1);
   int nrhs = 1;
   int info;

   LTL_ASSERT(A.length(2) == N, "Matrix not square");
   LTL_ASSERT(N == b.length(1), "The number of rows in A must equal the length of b!");

   ipiv.makeReference( MArray<int,1>(N));

   lapack_gesv_dispatch<T>::call(&N, &nrhs, A.data(), &N, ipiv.data(), b.data(), &N, &info);

   return (info==0);
}

/*!
 DGETRF - compute an LU factorization of a general M-by-N
      matrix A using partial pivoting with row interchanges

 DGETRF computes an LU factorization of a general M-by-N
      matrix A using partial pivoting with row interchanges.

      The factorization has the form
         A = P * L * U
      where P is a permutation matrix, L is lower triangular with
      unit diagonal elements (lower trapezoidal if m > n), and U
      is upper triangular (upper trapezoidal if m < n).

      This is the right-looking Level 3 BLAS version of the algorithm.
*/
template<typename T>
bool lapack_getrf( MArray<T,2>& A, MArray<int,1>& ipiv )
{
   int M = A.length(1);
   int N = A.length(2);
   ipiv.makeReference (MArray<int,1>( (M<N?M:N) ));
   int info;
   lapack_getrf_dispatch<T>::call( &M, &N, A.data(), &M, ipiv.data(), &info );
   return (info==0);
}

/*!
 GETRI computes the inverse of a matrix using the LU factorization computed by DGETRF.

 This method inverts U and then computes inv(A) by solving
 the system inv(A)*L = inv(U) for inv(A).

     A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
              On entry, the factors L and U from the factorization
              A = P*L*U as computed by DGETRF.  On exit, if INFO =
              0, the inverse of the original matrix A.

      IPIV    (input) INTEGER array, dimension (N)
              The pivot indices from DGETRF; for 1<=i<=N, row i of
              the matrix was interchanged with row IPIV(i).
*/
template<typename T>
bool lapack_getri( MArray<T,2>& A, MArray<int,1>& ipiv )
{
   int N = A.length(1);
   LTL_ASSERT(A.length(2) == N, "Matrix not square in getri");
   int Nwork = 16*N, info;
   MArray<T,1> work(16*N);
   lapack_getri_dispatch<T>::call( &N, A.data(), &N, ipiv.data(), work.data(), &Nwork, &info );
   return (info==0);
}

/*!
 *  GESV computes the solution to Nrhs systems of linear equations
 *     A * X = B,
 *  where A is an N-by-N matrix and X and B are N x Nrhs-matrices.
 *
 *  LU decomposition with partial pivoting and row interchanges is
 *  used to factor A as
 *     A = P * L * U,
 *  where P is a permutation matrix, L is unit lower triangular, and U is
 *  upper triangular.  The factored form of A is then used to solve the
 *  system of equations A * x = b.
 *
 *  The matrix A is replaced by L and U.
 *  The Matrix B is overwritten with the solutions of A x_i = b_i,
 *  where x_i and b_i are the column vectors of X and B
 */
template <typename T>
bool lapack_gesv( MArray<T,2>& A, MArray<T,2>& B, MArray<int,1> ipiv )
{
   int N = A.length(1);
   int nrhs = B.length(2);
   int info;

   LTL_ASSERT(A.length(2) == N, "Matrix not square in gesv");
   LTL_ASSERT(N == B.length(1), "The number of rows in A must equal the length of b!");

   ipiv.makeReference( MArray<int,1>(N));

   lapack_gesv_dispatch<T>::call(&N, &nrhs, A.data(), &N, ipiv.data(), B.data(), &N, &info);

   return (info==0);
}

/*!
 *  Purpose
 *  =======
 *
 *  SYEV computes all eigenvalues and, optionally, eigenvectors of a
 *  real symmetric matrix A.
 *
 */
template <typename T>
bool lapack_syev( const MArray<T,2>& A, MArray<T,1>& val, MArray<T,2>& vec )
{
   char jobz = 'V';
   char uplo = 'U';
   int N = A.length(1);
   int lda=N;
   int lwork = 3*N;
   int info;
   MArray<T,1> work(lwork);

   vec = A; // Input matrix will be overwritten with the eigenvectors, so need to copy.

   LTL_ASSERT(A.length(2) == N, "Matrix not square");
   LTL_ASSERT(val.length(1) == N, "Vector of eigenvalues must be equal to the side of the matrix");
   LTL_ASSERT(vec.isConformable(A), "Matrix of eigenvectors must have same shape as the matrix");

   lapack_syev_dispatch<T>::call ( &jobz, &uplo, &N, vec.data(), &lda, val.data(), work.data(), &lwork, &info );

   if (work(1) != lwork)
   {
     std::cout << "[LAPACK_GEEV] MESSAGE: Optimal lwork = " << work(1) << "\n";
   }

   return (info==0);
}

/*!
 *  Purpose
 *  =======
 *
 *  SYEV computes all eigenvalues and, optionally, eigenvectors of a
 *  real symmetric matrix A in banded storage (assuming superdiagonals
 *  are stored.
 *
 */
template <typename T>
bool lapack_sbev( const MArray<T,2>& AB, MArray<T,1>& val, MArray<T,2>& vec )
{
   char jobz = 'V';
   char uplo = 'U';
   int N = AB.length(2);
   int ldab = AB.length(1);
   int kd = ldab - 1;
   int ldv = N;
   int info;
   MArray<T,1> work(3*N-2);

   LTL_ASSERT(val.length(1) == N, "Vector of eigenvalues must be equal to the side of the matrix");
   LTL_ASSERT(vec.length(1) == N || vec.length(2) == N, "Eigenvectors matrix must be square and its sides must be equal to the number of columns of AB");

   lapack_sbev_dispatch<T>::call ( &jobz, &uplo, &N, &kd, AB.data(), &ldab, val.data(), vec.data(), &ldv,
				   work.data(), &info );

   return (info==0);
}

//@}
#ifdef HAVE_COMPLEX
# undef CFLOAT
# undef CDOUBLE
#endif

}


#endif // #ifndef __LTL_LAPACK_H__
