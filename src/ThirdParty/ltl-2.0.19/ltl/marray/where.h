/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: where.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
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

#ifndef __LTL_IN_FILE_MARRAY__
#error "<ltl/marray/where.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_WHERE__
#define __LTL_WHERE__

#include <ltl/config.h>

namespace ltl {

template<typename T,int N>
class MArray;

/*! \defgroup where Where Expressions

\ingroup marray_class

Index an arbitrary subset of elements of an \c MArray.

For example, Find all elemets of an \c MArray that are zero:

\code
   MArray<float,3> A(nx,ny,nz);
   A = ...
   IndexList<3> zeros = where( A==0.0f );
\endcode

We can now set all of these to 1:
\code
   A[zeros] = 1;   // could be any conformable 1-D expression.
\endcode

You can use \c operator()(IndexList&) instead of \c operator[](IndexList&) to
obtain the values of the \c MArry indexed by the \c IndexList as a 1D \c MArray:
\code
   MArray<float,1> B = A(someIndexList);
\endcode

In short:
\code
   IndexList<N> where( [some N-dim boolean expression] );
   MArray<T,N>::operator[]( IndexList& ) = expression;
   Marray<T,1> MArray<T,N>::operator()( IndexList& );
\endcode
*/

/// \cond DOXYGEN_IGNORE

//! Used to implement the function ltl::where() which returns an ltl::IndexList.
template<int N>
class IndexList
{
   public:
      typedef FixedVector<int,N> Index;
      typedef typename list<Index>::const_iterator const_iterator;

      IndexList()
            : size_(0)
      { }

      IndexList( const IndexList<N>& other )
            : indexList_(other.indexList_), size_(other.size_)
      { }

      void add( const Index& v )
      {
         indexList_.push_back( v );
         ++size_;
      }

      const_iterator begin() const
      {
         return indexList_.begin();
      }

      const_iterator end() const
      {
         return indexList_.end();
      }

      size_t size() const
      {
         return size_;
      }

   protected:
      list<Index> indexList_;
      int size_;
};


// dummy class for statements of the form
// A( IndexList ) = expr
// Need something to handle the assignment operator ...
template<typename T, int N>
class IndexRef
{
   public:
      IndexRef( const IndexList<N>& il, MArray<T,N>& A )
            : il_(il), A_(A)
      { }

      template<typename Expr>
      void operator=( ExprNode<Expr,1> expr )
      {
         LTL_ASSERT( il_.size() == (expr.shape())->nelements(),
                     "Assignment to IndexList from MArray of different length" );

         typename IndexList<N>::const_iterator i=il_.begin(), e=il_.end();
         while( i != e )
         {
            A_( *i ) = static_cast<T>(*expr);
            expr.advance();
            ++i;
         }
      }

      template<typename T2>
      void operator=( const MArray<T2,1>& expr )
      {
         LTL_ASSERT( il_.size() == expr.nelements(),
                     "Assignment to IndexList from MArray of different length" );

         typename IndexList<N>::const_iterator i=il_.begin();
         typename MArray<T2,1>::ConstIterator ie = expr.begin();
         while( !ie.done() )
         {
            A_( *i ) = static_cast<T>(*ie);
            ++ie;
            ++i;
         }
      }

      void operator=( const T expr )
      {
         typename IndexList<N>::const_iterator i=il_.begin(), e=il_.end();
         while( i != e )
         {
            A_( *i ) = expr;
            ++i;
         }
      }


   protected:
      const IndexList<N>& il_;
      MArray<T,N>& A_;
};


// Finally, the global \c where() function for \c ExprBase type arguments
//
template<typename T1, int N>
IndexList<N> where( const ExprBase<T1,N>& a )
{
   IndexList<N> l;

   T1& e = const_cast<T1&>(a.derived());
   IndexIter<typename ExprNodeType<T1>::expr_type, N> i( e.shape() );
   while( !i.done() )
   {
      if( *e )
         l.add( i.index() );
      e.advance();
      i.advance();
      if( i.needAdvanceDim() )
      {
         e.advanceDim();
         i.advanceDim();
      }
   }
   return l;
}

/// \endcond

}

#endif // __LTL_WHERE__
