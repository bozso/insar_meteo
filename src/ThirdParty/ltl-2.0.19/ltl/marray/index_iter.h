/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: index_iter.h 541 2014-07-09 17:01:12Z drory $
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
#error "<ltl/marray/index_iter.h> must be included via <ltl/marray.h>, never alone!"
#endif

#ifndef __LTL_INDEXITER__
#define __LTL_INDEXITER__

#include <ltl/config.h>

namespace ltl {

/*! \defgroup index_iter Index Iterator Expressions
 *
 *  \ingroup marray_class
 *
 *  Iterator object holding vector of index values for each position.
 *  In many situations one has to fill arrays with values that are
 *  expressed as functions of "coordinates" within the array, or more
 *  generally, as functions of the array's indices. Imagine setting up
 *  initial conditions for a PDE, e.g. a circular sine wave on a 2-d
 *  surface. For this purpose index iterators are provided. They behave
 *  much like normal iterators, but instead of returning the array's value
 *  they return the indices of the element they currently point to.
 *
 *  An index iterator can be obtained from an \c MArray by calling
 *  \c indexBegin():
 *
 *  \code
 *  MArray<T,N> A( ... );
 *  MArray<T,N>::IndexIterator i = A.indexBegin();
 *  \endcode
 *
 *  \code IndexIterator ltl::MArray<T,N>::indexBegin (); \endcode
 *  Return an \c IndexIterator for the current \c ltl::MArray.
 *
 *  An index iterator always iterates over the index ranges of the array
 *  it was created from. They are used in the same way as normal
 *  iterators, except that they have some additional methods and no
 *  \c operator*:
 *
 *  \code
 *  FixedVector I = i(); // get index
 *  int x = i(1);        // get index in first dimension
 *  int y = i(2);        // get index in second dimension
 *  \endcode
 *
 *  Mostly you will prefer to use the 'automatic' version of the index
 *  iterators directly in expressions rather than creating an index
 *  iterator by hand and writing out the loop, \ref array_expressions.
 *  These look like an ordinary function in the
 *  expression. The "function" simply evaluates to the index of the
 *  current element during elementwise evaluation of the expression, e.g.
 *
 *  \code
 *  MArray E(10,10);
 *  E = indexPos(E,1)==indexPos(E,2); // 10x10 unity matrix
 *  \endcode
 *
 *  While \c indexPos() evaluates to an \c int, there are also spcialized
 *  versions that return \c float and \c double values, \c indexPosFlt() and
 *  \c indexPosDbl(), respectively. These are provided for convenience to avoid
 *  having to use cast expressions frequently.
 *
 */

 /*! \ingroup index_iter
  *  Iterator object holding vector of index values for each position.
  *
  *  This class is also used for implementing ltl::IndexList and ltl::where().
  */
template<typename T, int N>
class IndexIter: public LTLIterator
{
   public:
      enum { numIndexIter = 1 };
      enum { numConvolution = 0 };
      enum { isVectorizable = 0};

      IndexIter( const MArray<T,N>& array );
      IndexIter( const Shape<N>* s );
      IndexIter( const IndexIter<T,N>& other )
            : first_(other.first_), last_(other.last_),
            pos_(other.pos_), done_(other.done_),
            shape_(other.shape_)
      { }

      // reset iterator (back to first element)
      void reset();

      inline IndexIter<T,N>& operator++()
      {
         advance(); // increment innermost loop

         if( N == 1 )  // will be const-folded away, dont worry ...
            return *this;

         if( pos_(1) <= last_(1) )
            // We hit this case almost all the time.
            return *this;


         advanceDim(); // hit end of row/columns/whatever, need to reset loops
         return *this;
      }

      IndexIter<T,N> operator++( int )
      {
         IndexIter<T,N> tmp (*this);
         ++(*this);
         return tmp;
      }

      inline void advance()
      {
         ++pos_(1);
      }

      inline void advance( const int i )
      {
         pos_(1) += i;
      }

      inline void advance( const int i, const int dim )
      {
         pos_(dim+1) += i;
      }

      inline void advanceWithStride1()
      {
         ++pos_(1);
      }

      FixedVector<int,N> readWithoutStride( const int i ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(1) += i;
         return tmp;
      }

      FixedVector<int,N> readWithStride( const int i ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(1) += i;
         return tmp;
      }

      FixedVector<int,N> readWithStride( const int i, const int dim ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(dim) += i;
         return tmp;
      }

      FixedVector<int,N> readAtOffsetDim( const int i, const int dim ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(dim) += i;
         return tmp;
      }

      FixedVector<int,N> readAtOffset( const int i ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(1) += i;
         return tmp;
      }

      FixedVector<int,N> readAtOffset( const int i, const int j ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(1) += i;
         tmp(2) += j;
         return tmp;
      }

      FixedVector<int,N> readAtOffset( const int i, const int j, const int k ) const
      {
         FixedVector<int,N> tmp = pos_;
         tmp(1) += i;
         tmp(2) += j;
         tmp(3) += k;
         return tmp;
      }

      inline int boundary_l(const int) const
      {
         return 0;
      }

      inline int boundary_u(const int) const
      {
         return 0;
      }

      void advanceDim();
      void advanceDim(const int cutDim);

      inline bool needAdvanceDim() const
      {
         if( N == 1 )
            return false;
         else
            return pos_(1) > last_(1);
      }

      bool done() const
      {
         if( N == 1 )
            return pos_(1) > last_(1);
         else
            return done_;
      }

      const FixedVector<int,N>& index() const
      {
         return pos_;
      }

      int index( const int dim ) const
      {
         LTL_ASSERT( (dim>0 && dim<=N), "Bad dimension "<<dim
                     <<" for IndexIterator<"<<N<<"!" );
         return pos_(dim);
      }

      const FixedVector<int,N>& operator*() const
      {
         return index();
      }

      const FixedVector<int,N>& operator()() const
      {
         return index();
      }

      int operator()( const int dim ) const
      {
         return index(dim);
      }

 #ifdef LTL_USE_SIMD
      typedef int vec_value_type;
      inline typename VEC_TYPE(int) readVec( const int i ) const
      {
         return vec_trait<int>::init(0); //(typename VEC_TYPE(int))(0);
      }

      inline void alignData( const int align )
      {
      }

      inline int getAlignment() const
      {
         return 0;
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return false;
      }
#endif

      void printRanges() const;

      const Shape<N>* shape() const
      {
         return shape_;
      }

      // we can't use loop collapsing, since we need to keep track
      // of all indices ...
      bool isStorageContiguous( void ) const
      {
         return N==1;
      }

      bool isStride1() const
      {
         return true;
      }

      bool isConformable( const Shape<N>& other ) const
      {
         return shape_->isConformable( other );
      }

   protected:
      FixedVector<int,N> first_;
      FixedVector<int,N> last_;
      FixedVector<int,N> pos_;
      bool done_;
      const Shape<N> *shape_;
};

/// \cond DOXYGEN_IGNORE
template<typename T, int N>
IndexIter<T,N>::IndexIter( const MArray<T,N>& A )
{
   // copy information to avoid dereferencing array all the time
   for( int i=1; i<=N; i++ )
   {
      last_(i) = A.maxIndex(i);
      pos_(i) = first_(i) = A.minIndex(i);
   }

   done_ = false;
   shape_ = A.shape();
}

template<typename T, int N>
IndexIter<T,N>::IndexIter( const Shape<N>* s )
{
   for( int i=1; i<=N; i++ )
   {
      last_(i) = s->last(i);
      pos_(i) = first_(i) = s->base(i);
   }

   done_ = false;
   shape_ = s;
}

// reset the iterator (back to first element)
//
template<typename T, int N>
void IndexIter<T,N>::reset()
{
   pos_ = first_;
   done_ = false;
}


template<typename T, int N>
void IndexIter<T,N>::advanceDim()
{
   // We've hit the end of a row/column/whatever.  Need to
   // increment one of the loops over another dimension.
   int j=2;
   for( ; j<=N; ++j )
   {
      ++pos_(j);
      if( pos_(j) <= last_(j) )
         break;
   }

   // are we finished?
   if ( j > N )
   {
      done_ = true;
      return;
   }

   // Now reset all the last pointers
   for (--j; j > 0; --j)
   {
      pos_(j) = first_(j);
   }
   return;
}

template<typename T, int N>
void IndexIter<T,N>::advanceDim(const int cutDim)
{
   // We've hit the end of a row/column/whatever.  Need to
   // increment one of the loops over another dimension.
   int j=2;
   for( ; j<=N; ++j )
   {
      if( j==cutDim+1 )
         continue;
      ++pos_(j);
      if( pos_(j) <= last_(j) )
         break;
   }

   // are we finished?
   if ( j > N )
   {
      done_ = true;
      return;
   }

   // Now reset all the last pointers
   for (--j; j > 0; --j)
   {
      pos_(j) = first_(j);
   }
   return;
}

template<typename T, int N>
void IndexIter<T,N>::printRanges() const
{
   cerr << "Ranges: ";
   for(int i=1; i<=N; i++)
      cerr << "(" << last_(i) - first_(i) + 1 << ")  ";
}


//
// -----------------------------------------------------------------
//



template<typename T, typename RetType, int N>
class IndexIterDimExpr : public ExprBase< IndexIterDimExpr<T,RetType,N>, N >, public LTLIterator
{
   public:
      typedef RetType value_type;
      enum { numIndexIter = IndexIter<T,N>::numIndexIter };
      enum { numConvolution = IndexIter<T,N>::numConvolution };
      enum { isVectorizable = IndexIter<T,N>::isVectorizable };

      IndexIterDimExpr( const IndexIter<T,N>& a, const int dim )
            : iter_(a), dim_(dim)
      { }

      IndexIterDimExpr( const IndexIterDimExpr<T,RetType,N>& other )
            : iter_(other.iter_), dim_(other.dim_)
      { }

      value_type operator*() const
      {
         return static_cast<value_type>(iter_.index(dim_));
      }

      void operator++()
      {
         ++iter_;
      }

      void advance()
      {
         iter_.advance();
      }

      void advance( const int i )
      {
         iter_.advance(i);
      }

      void advanceWithStride1()
      {
         iter_.advanceWithStride1();
      }

      value_type readWithStride( const int i ) const
      {
         return iter_.readWithStride(i)(dim_);
      }

      value_type readWithoutStride( const int i ) const
      {
         return iter_.readWithoutStride(i)(dim_);
      }

      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return iter_.readAtOffsetDim(i,dim)(dim_);
      }

      value_type readAtOffset( const int i ) const
      {
         return iter_.readAtOffset(i)(dim_);
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         return iter_.readAtOffset(i,j)(dim_);
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return iter_.readAtOffset(i,j,k)(dim_);
      }

      int boundary_l(const int dim) const
      {
         return iter_.boundary_l(dim);
      }

      int boundary_u(const int dim) const
      {
         return iter_.boundary_u(dim);
      }

      void advanceDim()
      {
         iter_.advanceDim();
      }

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      {
         return iter_.readVec(i);
      }

      inline void alignData( const int align )
      {
      }

      inline int getAlignment() const
      {
         return 0;
      }

      inline bool sameAlignmentAs( int p ) const
      {
         return iter_.sameAlignmentAs(p);
      }
#endif

      bool isStorageContiguous() const
      {
         return iter_.isStorageContiguous();
      }

      bool isStride1() const
      {
         return iter_.isStride1();
      }

      bool isConformable( const Shape<N>& other ) const
      {
         return iter_.isConformable( other );
      }

      const Shape<N>* shape() const
      {
         return iter_.shape();
      }

      void reset()
      {
         iter_.reset();
      }

   protected:
      IndexIter<T,N> iter_;
      const int dim_;
};

/// \endcond

/*! and the wrapper...
 use indexPosInt( A ) in an expression to refer to the current
 index in MArray A during expression evaluation.
*/

/*! \ingroup index_iter
  Use indexPosInt( A, i ) in an expression to refer to the i-th dimension's
  index in MArray A during expression evaluation. The index value is
  returned as an \c int.
  */
template<typename T1, int N>
inline ExprNode<IndexIterDimExpr<T1,int,N>, N>
indexPosInt( const MArray<T1,N>& a, const int dim )
{
   typedef IndexIterDimExpr<T1,int,N> ExprT;
   return ExprNode<ExprT,N>( ExprT(a.indexBegin(), dim) );
}
#ifdef __GCC__
#  pragma GCC poison indexPos
#endif

/*! \ingroup index_iter
  indexPosFlt() return the index value as a float.
  */
template<typename T1, int N>
inline ExprNode<IndexIterDimExpr<T1,float,N>, N>
indexPosFlt( const MArray<T1,N>& a, const int dim )
{
   typedef IndexIterDimExpr<T1,float,N> ExprT;
   return ExprNode<ExprT,N>( ExprT(a.indexBegin(), dim) );
}

/*! \ingroup index_iter
  indexPosDbl() returns the index value as a double.
  */
template<typename T1, int N>
inline ExprNode<IndexIterDimExpr<T1,double,N>, N>
indexPosDbl( const MArray<T1,N>& a, const int dim )
{
   typedef IndexIterDimExpr<T1,double,N> ExprT;
   return ExprNode<ExprT,N>( ExprT(a.indexBegin(), dim) );
}

}

#endif // __LTL_INDEXITER__

