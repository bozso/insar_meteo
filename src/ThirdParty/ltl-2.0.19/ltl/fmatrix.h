/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmatrix.h 562 2015-04-30 16:01:16Z drory $
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

#ifndef __LTL_FMATRIX__
#define __LTL_FMATRIX__


// ====================================================================

// little trick to avoid files being included separetely
#define __LTL_IN_FILE_FMATRIX__
// this is #undef'd at the end of this file ...

// ====================================================================

#include <ltl/config.h>
#include <ltl/misc/mdebug.h>
#include <ltl/misc/type_name.h>

#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cmath>
#ifdef LTL_COMPLEX_MATH
#  include <complex>
   using std::complex;
#endif

using std::ostream;
using std::endl;
using std::setw;

#include <ltl/fvector.h>

// ====================================================================

#include <ltl/misc/staticinit.h>
#include <ltl/fmatrix/fmiter.h>
#include <ltl/fmatrix/fmexpr.h>
#include <ltl/fmatrix/fmtloops.h>
#include <ltl/fmatrix/fmexpr_ops.h>
#include <ltl/fmatrix/fmbool.h>
#include <ltl/fmatrix/fmtranspose.h>
#include <ltl/fmatrix/fmatvec.h>
#include <ltl/fmatrix/fmatmat.h>

// ====================================================================

namespace ltl {

//! Matrix with dimensions known at compile time.
/*! \ingroup fvector_fmatrix

  Indices are 1-based. The \c FMatrix does not require more storage
  than is necessary to hold the MxN elements.
  STL-compatible iterators and types.
*/
template<class T, int M, int N>
class FMatrix
{
      friend class FMIter<T,M,N>;
      friend class FMIterConst<T,M,N>;
      friend class ListInitializationSwitch< FMatrix<T,M,N> >;

   public:
      //@{
      //! STL-compatible type definitions
      typedef T              value_type;
      typedef FMIter<T,M,N>  iterator;
      typedef FMIterConst<T,M,N>       const_iterator;
      typedef T&             reference;
      typedef const T&       const_reference;
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;

      typedef FVector<T,M,N>   ColumnVector;
      typedef FVector<T,N,1>   RowVector;
      typedef FVector<T,M,N+1> TraceVector;
      //@}

      //! Used by expression engine to decide which loops to unroll.
      // (See LTL_TEMPLATE_LOOP_LIMIT)
      enum { static_size = 1 };

      //@{
      /*! Functions needed for STL container conformance */

      //! STL return \c M*N
      static size_type size()
      {
         return M*N;
      }

      //! STL \c empty(). Always false.
      static bool empty()
      {
         return false;
      }

      //! STL: Maximum capacity. Always \c ==size()
      static size_type max_size()
      {
         return M*N;
      }
      //@}

      //! default constructor
      FMatrix()
      { }

      //! default destructor
      ~FMatrix()
      { }

      //! copy constructor: copy elements of \c other.
      FMatrix( const FMatrix<T,M,N>& other );

      //! copy contents of memory pointed to by \c t.
      FMatrix( const T* t );

      //! fill with value \c t.
      FMatrix( const T t );

      //! construct from expression.
      template<class Expr>
      FMatrix( const FMExprNode<Expr,M,N>& e );

      //! Initialize with list of values or single value.
      /*!
       * Assign values through initialization list.
       * A bit more comlicated since we have to discriminate between
       * A = 3; and A = 1, 2, 3, 4;
       * which is done using ListInitializationSwitch which either calls
       * \c ListInitializer or \c FVector::fill().
       */
      ListInitializationSwitch< FMatrix<T,M,N> > operator=( T x )
      {
         return ListInitializationSwitch< FMatrix<T,M,N> >( *this, x );
      }

      //! return length of container (M*N).
      int length() const
      {
         return M*N;
      }

      //! return length of container (M*N).
      int nelements() const
      {
         return M*N;
      }

      //! return lowest index of dimension \c dim. Always 1.
      int minIndex( const int dim ) const
      {
         return 1;
      }

      //! return highest index of dimension \c dim. Returns M or N.
      int maxIndex( const int dim ) const
      {
         if( dim == 1 ) return M;
         return N;
      }

      //@{
      //! 1-based access to elements.
      T  operator()( const int i, const int j ) const
      {
         LTL_ASSERT( i>0 && i<=M, "Index 1 out of bounds in FMatrix. Index : "
                     << i << ", size "<<M );
         LTL_ASSERT( j>0 && j<=N, "Index 2 out of bounds in FMatrix. Index : "
                     << j << ", size "<<N );
         return __data_[(i-1)*N+(j-1)];
      }

      //! 1-based access to elements.
      T& operator()( const int i, const int j )
      {
         LTL_ASSERT( i>0 && i<=M,
                     "Index 1 out of bounds in FMatrix(). Index : "
                     << i << ", size "<<M );
         LTL_ASSERT( j>0 && j<=N,
                     "Index 2 out of bounds in FMatrix(). Index : "
                     << j << ", size "<<N );
         return __data_[(i-1)*N+(j-1)];
      }
      //@}

      //@{
      //! Direct zero-based access to the (linear) block of memory
      T  operator[]( const int i ) const
      {
         LTL_ASSERT( i>=0 && i<M*N,
                     "Direct index out of bounds in FMatrix[]. Index : "
                     <<i<<" Range 0-"<<length()-1 );
         return __data_[i];
      }

      //! Direct zero-based access to the (linear) block of memory
      T& operator[]( const int i )
      {
         LTL_ASSERT( i>=0 && i<M*N,
                     "Direct index out of bounds in FMatrix[]. Index : "
                     <<i<<" Range 0-"<<length()-1 );
         return __data_[i];
      }
      //@}

      //! Return an ltl::FVector object REFERENCEING the column vector \c col.
      ColumnVector col( const int col )
      {
         LTL_ASSERT( col>0 && col<=N,
                     "Column index out of bounds in FMatrix.col(). Index : "
                     <<col<<" Range 1-"<<N );
         return ColumnVector( __data_-1+col );
      }

      //! Return an ltl::FVector object REFERENCEING the row vector \c row.
      RowVector row( const int row )
      {
         LTL_ASSERT( row>0 && row<=M,
                     "Row index out of bounds in FMatrix.row(). Index : "
                     <<row<<" Range 1-"<<M );
         return RowVector( __data_+(row-1)*N );
      }

      //! Return an ltl::FVector object REFERENCEING the trace vector.
      TraceVector traceVector()
      {
         LTL_ASSERT( M==N,
                     "Trace only defined for a square Matrix");
         return TraceVector( __data_ );
      }

      //! Return a pointer to the data.
      T* data()
      {
         return __data_;
      }

      //! Return a const pointer to the data.
      const T* data() const
      {
         return __data_;
      }

      //! return an iterator pointing to the first element.
      iterator begin()
      {
         return FMIter<T,M,N>( *this );
      }

      //! return a const iterator pointing to the first element.
      const_iterator begin() const
      {
         return FMIterConst<T,M,N>( *this );
      }

      //! fill with value \c x.
      void fill( const T x );

      //@{
      //! \c operatorX= for expression rhs.
      template<class Expr>
      FMatrix<T,M,N>& operator=( const FMExprNode<Expr,M,N>& e );

      template<class Expr>
      FMatrix<T,M,N>& operator+=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator-=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator*=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator/=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator%=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator^=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator&=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator|=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator<<=( const FMExprNode<Expr,M,N>& e );
      template<class Expr>
      FMatrix<T,M,N>& operator>>=( const FMExprNode<Expr,M,N>& e );
      //@}

      //@{
      //! operatorX for \c FMatrix rhs.
      template<class T2>
      FMatrix<T,M,N>& operator=( const FMatrix<T2,M,N>& v );

      FMatrix<T,M,N>& operator=( const FMatrix<T,M,N>& v );

      template<class T2>
      FMatrix<T,M,N>& operator+=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator-=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator*=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator/=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator%=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator^=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator&=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator|=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator<<=( const FMatrix<T2,M,N>& v );
      template<class T2>
      FMatrix<T,M,N>& operator>>=( const FMatrix<T2,M,N>& v );
      //@}

      //@{
      //! operatorX= for scalar rhs.
      FMatrix<T,M,N>& operator+=( const T t );
      FMatrix<T,M,N>& operator-=( const T t );
      FMatrix<T,M,N>& operator*=( const T t );
      FMatrix<T,M,N>& operator/=( const T t );
      FMatrix<T,M,N>& operator%=( const T t );
      FMatrix<T,M,N>& operator^=( const T t );
      FMatrix<T,M,N>& operator&=( const T t );
      FMatrix<T,M,N>& operator|=( const T t );
      FMatrix<T,M,N>& operator<<=( const T t);
      FMatrix<T,M,N>& operator>>=( const T t);
      //@}

      //! Swap the values in two row vectors.
      void swapRows(const int row1, const int row2);

      //! Swap the values in two column vectors.
      void swapCols(const int col1, const int col2);

   protected:
      T __data_[M*N];   //!< Storage for the matrix elements.
};


/*! \relates ltl::FMatrix
 *
 *  Write ltl::FVector to ascii stream. Compatible with
 *  \c opertor>>.
 */
template<class T, int M, int N>
ostream& operator<<( ostream& os, const FMatrix<T,M,N>& A )
{
   os << "FMatrix< "<<LTL_TYPE_NAME(T)<<", "<<M<<", "<<N<<" >" << endl;
   os << "[";
   for (int i=A.minIndex(1); i <= A.maxIndex(1); ++i)
   {
      os << "[ ";
      for (int j=A.minIndex(2); j <= A.maxIndex(2); ++j)
      {
         os << A(i,j) << " ";
      }
      os << "]";
      if( i<A.maxIndex(1) )
         os << endl << " ";
   }

   os << "]";
   return os;
}


/*! \relates ltl::FMatrix
 *
 *  Read ltl::FMatrix from ascii stream. Compatible with
 *  \c operator<<.
 */
template<class T, int M, int N>
istream& operator>>( istream& is, FMatrix<T,M,N>& A )
{
   T t;
   string tmp;

   is >> tmp;

   is >> tmp;
   while( tmp[tmp.length()-1] != '[' && !is.bad() && !is.eof() )
      is >> tmp;

   if( is.bad() || is.eof() )
      throw( IOException( "Format error while reading FVector: '[' expected, got "+tmp ) );

   int n=0;

   for( int i=A.minIndex(1); i<=A.maxIndex(1); ++i )
   {
      for( int j=A.minIndex(2); j<=A.maxIndex(2); ++j )
      {
         is >> t;
         A(i,j) = t;
         ++n;
         if( is.bad() )
            throw( IOException( "Format error while reading FMatrix!" ) );
      }
      is >> tmp;

      if( tmp[tmp.length()-1] != ']' )
         throw( IOException( "Format error while reading FMatrix: ']' expected, got"+tmp ) );

      is >> tmp;
      if( n==M*N ) continue;
      if( tmp[tmp.length()-1] != '[' || is.bad() || is.eof() )
         throw( IOException( "Format error while reading FVector: '[' expected, got "+tmp ) );
   }
   return is;
}



}

#include <ltl/fmatrix/fmatrix_ops.h>
#include <ltl/fmatrix/fmatrix_methods.h>

#undef __LTL_IN_FILE_FMATRIX__

#endif //__LTL_FMATRIX__
