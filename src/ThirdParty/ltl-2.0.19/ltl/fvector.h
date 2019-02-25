/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvector.h 562 2015-04-30 16:01:16Z drory $
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

#ifndef __LTL_FVECTOR__
#define __LTL_FVECTOR__


// ====================================================================

// little trick to avoid files being included separetely
#define __LTL_IN_FILE_FVECTOR__
// this is #undef'd at the end of this file ...

// ====================================================================

#include <ltl/config.h>
#include <ltl/misc/mdebug.h>
#include <ltl/misc/exceptions.h>
#include <ltl/misc/type_name.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <cstddef>
#include <cmath>
#ifdef LTL_COMPLEX_MATH
#  include <complex>
   using std::complex;
#endif

using std::ostream;
using std::istream;
using std::string;
using std::endl;
using std::setw;

// ====================================================================

namespace ltl {
template<class T, int N, int S> class FVector;
}

#include <ltl/misc/staticinit.h>

#include <ltl/fvector/fvmemory.h>
#include <ltl/fvector/fviter.h>
#include <ltl/fvector/fvexpr.h>
#include <ltl/fvector/fvtloops.h>
#include <ltl/fvector/fvexpr_ops.h>
#include <ltl/fvector/fvbool.h>
#include <ltl/fvector/fvdot.h>

namespace ltl {

// ====================================================================

//! Vector whose length is known at compile time.

/*! \ingroup fvector_fmatrix

   Can either have its own memory region or reference foreign
   memory, for example when representing a column-vector of a matrix

   The template parameter S is a 'stride' for the foreign memory: when
   we reference a column vector we need a stride of M (if the matrix
   is MxN). Memory handling is encapsulated in the class FVMemory.
   To make life easier for the compiler/optimizer, there
   is a specialization of FVMemory with _no_ stride at all and
   an own new embedded block of memory. This is indicated
   by S=0, which is also the default case.

   \c FVector does not require more memory than is necessary
   to store the N elements of the vector if S=0. If S>0, the size is exactly
   \c sizeof(T*).

   \c FVector provides operator() for 1-based access and operator[]
   for 0-based access. A full expression-templated engine for evaluating
   expressions involving ltl::FVectors is provided, as well as
   dot products between ltl::FVector and ltl::FMatrix objects, and
   scalar-valued reductions.

   STL-compatible iterators and types.
*/
template<class T, int N, int S=0>
class FVector : public FVMemory<T,N,S>
{
      friend class FVIter<T,N,S>;
      friend class FVIterConst<T,N,S>;
      friend class ListInitializationSwitch<FVector<T,N,S> >;

   public:
      //@{
      //! STL-compatible type definitions
      typedef T                   value_type;
      typedef FVIter<T,N,S>       iterator;
      typedef FVIterConst<T,N,S>  const_iterator;
      typedef T*                  pointer;
      typedef const T*            const_pointer;
      typedef T&                  reference;
      typedef const T&            const_reference;
      typedef std::size_t         size_type;
      typedef std::ptrdiff_t      difference_type;
      //@}

      //! Used by expression engine to decide which loops to unroll.
      // (See LTL_TEMPLATE_LOOP_LIMIT)
      enum { static_size = 1 };

      //@{
      /*! Functions needed for STL container conformance */

      //! STL return the length of the vector
      inline static size_type size()
      {
         return N;
      }

      //! STL \c empty(). Always false.
      inline static bool empty()
      {
         return false;
      }

      //! STL: Maximum capacity. Always \c ==size()
      inline static size_type max_size()
      {
         return N;
      }
      //@}

      //! default constructor
      FVector()
      { }

      //! destructor
      ~FVector()
      { }

      //! copy constructor
      FVector( const FVector<T,N,S>& other );

      /*! constructor taking a pointer to foreign memory.
       *  If \c S==0, copy, else reference the foreign memory.
       */
      FVector( T* const a );

      //! fill with value \c t.
      FVector( const T t );

      //! construct from expression
      template<class Expr>
      FVector( const FVExprNode<Expr,N>& e );

      //! Initialize with list of values or single value.
      /*!
       * Assign values through initialization list.
       * A bit more comlicated since we have to discriminate between
       * A = 3; and A = 1, 2, 3, 4;
       * which is done using ListInitializationSwitch which either calls
       * \c ListInitializer or \c FVector::fill().
       */
      ListInitializationSwitch< FVector<T,N,S> > operator=( T x )
      {
         return ListInitializationSwitch< FVector<T,N,S> >( *this, x );
      }

      /*!
       * operator[] and operator() inherited from FVMemory
       */

      //! return length of vector.
      inline static int length()
      {
         return N;
      }

      //! return length of vector.
      inline static int nelements()
      {
         return N;
      }

      //! lowest possible index, always one.
      inline static int minIndex()
      {
         return 1;
      }

      //! highest possible index, always N.
      inline static int maxIndex()
      {
         return N;
      }

      //! return an iterator pointing to the first element.
      inline iterator begin()
      {
         return FVIter<T,N,S>( *this );
      }

      //! return a const iterator pointing to the first element.
      inline const_iterator begin() const
      {
         return FVIterConst<T,N,S>( *this );
      }

      //! return an iterator pointing past the last element.
      inline iterator end()
      {
         return FVIter<T,N,S>( *this, fviter_end_tag() );
      }

      //! return a const iterator pointing past the last element.
      inline const_iterator end() const
      {
         return FVIterConst<T,N,S>( *this, fviter_end_tag() );
      }

      //! fill with value \c x.
      void fill( const T x );

      //@{
      //! \c operatorX= for expression rhs.
      template<class Expr>
      FVector<T,N,S>& operator=( const FVExprNode<Expr,N>& e );

      template<class Expr>
      FVector<T,N,S>& operator+=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator-=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator*=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator/=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator%=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator^=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator&=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator|=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator<<=( const FVExprNode<Expr,N>& e );
      template<class Expr>
      FVector<T,N,S>& operator>>=( const FVExprNode<Expr,N>& e );
      //@}

      //@{
      //! operatorX for \c FVector rhs.
      template<class T2, int S2>
      FVector<T,N,S>& operator=( const FVector<T2,N,S2>& v );

      FVector<T,N,S>& operator=( const FVector<T,N,S>& v );

      template<class T2, int S2>
      FVector<T,N,S>& operator+=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator-=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator*=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator/=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator%=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator^=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator&=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator|=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator<<=( const FVector<T2,N,S2>& v );
      template<class T2, int S2>
      FVector<T,N,S>& operator>>=( const FVector<T2,N,S2>& v );
      //@}

      //@{
      //! operatorX= for scalar rhs.
      FVector<T,N,S>& operator+=( const T t );
      FVector<T,N,S>& operator-=( const T t );
      FVector<T,N,S>& operator*=( const T t );
      FVector<T,N,S>& operator/=( const T t );
      FVector<T,N,S>& operator%=( const T t );
      FVector<T,N,S>& operator^=( const T t );
      FVector<T,N,S>& operator&=( const T t );
      FVector<T,N,S>& operator|=( const T t );
      FVector<T,N,S>& operator<<=( const T t);
      FVector<T,N,S>& operator>>=( const T t);
      //@}

      //! swap values with \c other.
      template<class T2, int S2>
      void swap( FVector<T2, N, S2>& other );
};

/*! \relates ltl::FVector
 *
 *  Write ltl::FVector to ascii stream. Compatible with
 *  \c opertor>>.
 */
template<class T, int N, int S>
ostream& operator<<(ostream& os, const FVector<T,N,S>& x)
{
   os << "FVector< "<<LTL_TYPE_NAME(T)<<","<<N<<","<<S<<" >" << endl;
   os << " [ ";
   for (int i=x.minIndex(); i <= x.maxIndex(); ++i)
   {
      os << x(i) << " ";
      if (!((i+1-x.minIndex())%9))
         os << endl << "  ";
   }
   os << " ]";
   return os;
}

/*! \relates ltl::FVector
 *
 *  Read ltl::FVector from ascii stream. Compatible with
 *  \c operator<<.
 */
template<class T, int N, int S>
istream& operator>>( istream& is, FVector<T,N,S>& x )
{
   T t;
   string tmp;
   is >> tmp;
   while( tmp[tmp.length()-1] != '[' && !is.bad() && !is.eof() )
      is >> tmp;

   if( is.bad() || is.eof() )
      throw( IOException( "Format error while reading FVector: '[' expected, got "+tmp ) );

   for( int i=x.minIndex(); i<=x.maxIndex(); ++i )
   {
      is >> t;
      if( is.bad() )
         throw( IOException( "Format error while reading FVector!" ) );
      x(i) = t;
   }

   is >> tmp;
   if( tmp[tmp.length()-1] != ']' )
         throw( IOException( "Format error while reading FVector: ']' expected, got "+tmp ) );

   return is;
}


}

#include <ltl/fvector/fvector_ops.h>
#include <ltl/fvector/fvector_methods.h>

#undef __LTL_IN_FILE_FVECTOR__

#endif //__LTL_FVECTOR__
