/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marray_iter.h 541 2014-07-09 17:01:12Z drory $
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
#error "<ltl/marray/marray_iter.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_ITER__
#define __LTL_ITER__

#include <ltl/config.h>

namespace ltl {

template<typename T, int N>
class MArray;

template<int N>
class Shape;

struct LTLIterator // solely to keep all iterators together in code browsers
{ };

/*! \file marray_iter.h
  General ltl::MArray iterators: a const and a non-const version.
  These iterators can deal with any array geometry/topology.
*/

/// \cond DOXYGEN_IGNORE
struct _iter_end_tag
{ };
/// \endcond


/*! \defgroup marray_iter MArray iterators
 *
 *  \ingroup marray_class
 *
 *  Iterators are "generalized pointers" to the elements of an array of
 *  arbitrary geometry, providing two operations, namely dereferencing via
 *  \c operator* (returning the value of the element currently pointed
 *  to or a reference to it for use on the rhs) and moving the "pointer"
 *  to the next element via \c operator++. In addition, these classes
 *  provide the necessary infrastructure to efficiently loop
 *  over arbitrary array geometries. This is used by the expression template
 *  engine and the vectorizer.
 *
 *  As a neat extra feature, the iterators are compatible with those known
 *  from the Standard Template Library (STL). So you can use the STL
 *  algorithms on \c MArrays. The only thing to know here is that the
 *  standard iterators in the LTL are only models of \c forward_iterators.
 *
 *  We provide a second set of iterators which are true
 *  \c random_access_iterators, but only under the precondition that
 *  the associated \c ltl::MArray object has a contiguous memory layout
 *  (which is always true for newly allocated arrays and for suitable
 *  subarrays or slices). The random-access iterators are then simply
 *  pointers to the array's data, aka objects of type \c T*. These are
 *  obtained by calling MArray::beginRA() and MArray::endRA().
 *
 */


/*! \ingroup marray_iter
 *
 *  Const iterator object for ltl::MArrays. Conforms to std::forward_iterator.
 *  The non-const version is implemented below by inheriting from this
 *  class.
 *
 */
template<typename T, int N>
class MArrayIterConst : public LTLIterator
{
   public:
      //@{
      //! \c std::iterator typedefs
      typedef std::forward_iterator_tag             iterator_category;

      typedef int                                   difference_type;
      typedef typename MArray<T,N>::value_type      value_type;
      typedef typename MArray<T,N>::const_reference const_reference;
      typedef typename MArray<T,N>::const_pointer   const_pointer;
      typedef typename MArray<T,N>::reference       reference;
      typedef typename MArray<T,N>::pointer         pointer;
      //@}

      //@{
      //! These define constants that are used by the expression template engine
      //  to determine the most efficient mode of looping using this iterator.
      enum { dims=N };
      enum { numIndexIter = 0 };
      enum { numConvolution = 0 };
      enum { isVectorizable = 1 };
      //@}

      //! Construct an \c iterator for a given ltl::MArray. This will rarely be called by a user
      // directly. Rather, use MArray::begin().
      MArrayIterConst( const MArray<T,N>& array );

      //! Construct an \c end-iterator for a given ltl::MArray. This will rarely be called by a user
      // directly. Rather, use MArray::end().
      MArrayIterConst( const MArray<T,N>& array,
                       const _iter_end_tag& E )
         : stride_(array.stride(1))
      { 
         LTL_UNUSED(E);
         
         // end iterator
         if( N==1 )
            data_ = last_[0] = array.data() + array.nelements()*stride_;
         else
            data_ = 0;
      }

      //! copy constructor
      MArrayIterConst( const MArrayIterConst<T,N>& other )
            : data_(other.data_), array_(other.array_), stride_(other.stride_)
      {
#ifdef EXPR_CTOR_DEBUG
      cout << "MArrayIterConst copy ctor" << endl;
#endif
         for( int i=0; i<N; ++i )
         {
            length_[i]  = other.length_[i];
            strides_[i] = other.strides_[i];
            stack_[i]   = other.stack_[i];
            last_[i]    = other.last_[i];
         }
      }

      //! Reset the iterator back to the first element
      void reset();

      //! Dereference the iterator object. Return the element pointed to.
      inline value_type  operator*() const
      {
         LTL_ASSERT( !done(), "Dereferencing end iterator" );
         return *data_;
      }

      //! Move the iterator to the next object.
      /*!
       *  This method will almost never be used. Instead, more efficient versions will be used by
       *  the expression template engine to move iterators forward.
       *
       *  This implementation of \c operator++() will check for the end condition every time it is
       *  called. When we evaluate an expression, all iterators involved in the expression point to
       *  MArrays or expressions of the same geometry, so checking for the end condition on ONE of the
       *  iterators is enough. We also know the length of the dimension(s), so we can optimize the
       *  increment past the end of one dimension (where we have to reset this dimension and use
       *  a different stride to increment the next-outer dimension).
       *  See below and implementation of evaluation methods in ltl/marray/eval.h.
       *
       */
      inline MArrayIterConst<T,N>& operator++()
      {
         LTL_ASSERT( !done(), "Incrementing past end iterator" );
         advance(); // increment innermost loop

         if( N == 1 )  // will be const-folded away, dont worry ...
            return *this;

         if( data_ != last_[0] )
         {
            // We hit this case almost all the time.
            return *this;
         }

         advanceDim(); // hit end of row/columns/whatever, need to reset loops
         return *this;
      }

      //! Postfix \c operator++().
      void operator++( int )
      {
         MArrayIterConst<T,N> tmp (*this);
         ++(*this);
         return tmp;
      }

      //@{

      //! Provide separate versions of advance() for incrementing the innermost
      //  loop and needAdvanceDim()/advanceDim() for incrementing an outer loop.

      /*!
       *  When evaluating an expression involving more than one iterator
       *  all terms MUST have the same geometry, the loop structures are
       *  identical. It's therefore sufficient to check the end of loop condition
       *  and the end of one dimension on one of the iterators ...
       *  All others are then 'remote controlled' via these methods methods.
       *  See implementation of evaluation methods in ltl/marray/eval.h.
       */
      //! Unconditionally increase the data pointer by one innermost stride.
      inline void advance()
      {
         data_ += stride_;
      }

      //! Unconditionally increase the data pointer by N innermost strides.
      inline void advance( int n )
      {
         data_ += n * stride_;
      }

      //! Unconditionally increase the data pointer by N along dimension dim (used for partial reductions).
      inline void advance( int n, int dim )
      {
         data_ += n * strides_[dim];
      }

      //! Unconditionally increase the data pointer by one.
      inline void advanceWithStride1()
      {
         ++data_;
      }

      //! Check if we have reached the end of the innermost dimension.
      inline bool needAdvanceDim() const
      {
         /*
          * For 1-d arrays, we can avoid calling needAdvanceDim() since it's
          * equivalent to the done condition
          */
         if( N == 1 )
            return false;
         else
            return data_ == last_[0];
      }

      //! Advance the iterator past the end of one "line" (the end of the innermost dimension).
      void advanceDim();

      //! Advance the iterator past the end of one "line", ignoring the dimension cutDim (used in partial reductions).
      void advanceDim( const int cutDim );
      //@}

      //@{
      /*!  These methods are for implementing loop unrolling and vectorization
       *   for the efficient evaluation of expression templates.
       */

      //!  Read the data value at the current location + i (optimized for stride 1).
      inline value_type readWithoutStride( const int i ) const
      {
         return data_[i];
      }

      //!  Read the current data value at the current location + i*stride
      inline value_type readWithStride( const int i ) const
      {
         return data_[i*stride_];
      }

      //!  Read the data value at position i along dimension dim (used in partial reductions)
      inline value_type readWithStride( const int i, const int dim ) const
      {
         return data_[i*strides_[dim]];
      }

      //! Read the data value at offset i along dim dim (used in convolutions)
      inline value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return data_[i*strides_[dim-1]];
      }

      //! Read the data value at offset i along dim 1 (used in convolutions)
      inline value_type readAtOffset( const int i ) const
      {
         return data_[i*stride_];
      }

      //! Read the data value at offset i along dim 1 and j along dim 2 (used in convolutions)
      inline value_type readAtOffset( const int i, const int j ) const
      {
         return data_[i*strides_[0] + j*strides_[1]];
      }

      //! Read the data value at offset i along dim 1, j along dim 2, and k along dim 3 (used in convolutions)
      inline value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return data_[i*strides_[0] + j*strides_[1] + k*strides_[2]];
      }

      //! leave boundary around the lower edges of the array (for convolutions)
      int boundary_l(const int dim) const
      {
         LTL_UNUSED(dim);
         return 0;
      }

      //! leave boundary around the upper edges of the array (for convolutions)
      int boundary_u(const int dim) const
      {
         LTL_UNUSED(dim);
         return 0;
      }

#ifdef LTL_USE_SIMD
      //!  The \c value_type if we are vectorizing
      typedef value_type vec_value_type;

      //!  Read the current vector data value at the current location + i
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      {
         return ((typename VEC_TYPE(value_type) *) data_)[i];
      }

      //!  Move the data pointer to a natural alignment boundary for our vector type.
      inline void alignData( const int align )
      {
         data_ += align;
      }

      //!  Return the misalignment relative to a natural vector alignment boundary. Assumes a vector length of 128 bits.
      inline int getAlignment() const
      {
         return (int)((long)data_ & 0x0FL);
      }

      //!  Check whether we have the same alignment realtive to a natural vector boundary. Assumes a vector length of 128 bits.
      inline bool sameAlignmentAs( const int p ) const
      {
         return ( p == ((long)data_ & 0x0FL) );
      }
#endif
      //@}

      //! \c true if iterators share the same data pointer
      bool operator==( const MArrayIterConst<T,N>& other ) const
      {
         return data_ == other.data_;
      }

      //! \c true if iterators do not share the same data pointer
      bool operator!=( const MArrayIterConst<T,N>& other ) const
      {
         return data_ != other.data_;
      }

      //! True if we point past the end, i.e. we are equal to the end iterator.
      inline bool done() const
      {
         if( N == 1 )  // if() will be const-folded away ...
            return data_ == last_[0];
         else
            return !data_;
      }

      //! \c true if we have the same array geometry as that of \c other,
      //  defined as the lengths of each dimension being equal to that of the same dimension in \c other.
      //  The base indices do not matter.
      bool isConformable( const Shape<N>& other ) const
      {
         return shape()->isConformable( other );
      }

      //! Pretty print the geometry.
      void printRanges() const;

      //! Return our \c shape
      const Shape<N>* shape() const
      {
         return array_->shape();
      }

      //! \c true if the memory we are pointing to has contiguous memory layout.
      //  Used in choosing the optimal looping technique when evaluating expressions
      //  (loop collapsing).
      bool isStorageContiguous() const
      {
         return shape()->isStorageContiguous();
      }

      //! \c true if our innermost stride is 1
      bool isStride1() const
      {
         return stride_==1;
      }

      //! Return the current data pointer.
      value_type* data() const
      {
         return data_;
      }

   protected:
#ifndef __xlC__
      value_type*       restrict_ data_;
#else
      value_type*       data_;
#endif
      const MArray<T,N> *array_;
      value_type        *stack_[N], *last_[N];
      int               strides_[N], length_[N];
      const int         stride_;
};

/// \cond DOXYGEN_IGNORE
//@{
//! The constructor implementation.
//
template<typename T, int N>
MArrayIterConst<T,N>::MArrayIterConst( const MArray<T,N>& array )
      : data_(array.data()), array_(&array),
      stride_(array.stride(1))
{
   LTL_ASSERT( data_!=NULL, "MarrayIter involving uninitialized MArray!" );
   // copy information to avoid dereferencing array all the time
   for( int i=0; i<N; i++ )
   {
      strides_[i] = array.stride(i+1);
      length_[i] = array.length(i+1);
      stack_[i] = data_;
      last_[i] = data_ + length_[i] * strides_[i];
   }
}


//! Reset the iterator (back to first element).
//
template<typename T, int N>
void MArrayIterConst<T,N>::reset()
{
   // reset data
   data_ = array_->data();

   // init stack and upper limits
   for( int i=0; i<N; i++ )
   {
      stack_[i] = data_;
      last_[i] = data_ + length_[i] * strides_[i];
   }
}


//! Reset stacks after we've hit the end of a dimension.
//
template<typename T, int N>
void MArrayIterConst<T,N>::advanceDim()
{
   // We've hit the end of a row/column/whatever.  Need to
   // increment one of the loops over another dimension.
   int j=1;
   for( ; j<N; ++j )
   {
      data_ = stack_[j];
      data_ += strides_[j];
      if( data_ != last_[j] )
         break;
   }

   // are we finished?
   if ( j == N )
   {
      // Setting data_ to 0 indicates the end of the array has
      // been reached, and will match the end iterator.
      data_ = 0;
      return;
   }

   stack_[j] = data_;

   // Now reset all the last pointers
   for (--j; j >= 0; --j)
   {
      stack_[j] = data_;
      last_[j] = data_ + length_[j] * strides_[j];
   }
}

//! Reset stacks after we've hit the end of a dimension. Ignore dimension \c cutDim used in partial reductions.
//
template<typename T, int N>
void MArrayIterConst<T,N>::advanceDim( const int cutDim )
{
   // We've hit the end of a row/column/whatever.  Need to
   // increment one of the loops over another dimension.
   int j=1;
   for( ; j<N; ++j )
   {
      if( j == cutDim )
         continue;
      data_ = stack_[j];
      data_ += strides_[j];
      if( data_ != last_[j] )
         break;
   }

   // are we finished?
   if ( j == N )
   {
      // Setting data_ to 0 indicates the end of the array has
      // been reached, and will match the end iterator.
      data_ = 0;
      return;
   }

   stack_[j] = data_;

   // Now reset all the last pointers
   for (--j; j >= 0; --j)
   {
      stack_[j] = data_;
      last_[j] = data_ + length_[j] * strides_[j];
   }
}

template<typename T, int N>
void MArrayIterConst<T,N>::printRanges() const
{
   cerr << "Ranges: ";
   for(int i=0; i<N; i++)
      cerr << "(" << length_[i] << ")  ";
}
//@}
/// \endcond


/*! \ingroup marray_iter
 *
 *  Non-const iterator object for ltl::MArrays. Conforms to std::forward_iterator.
 *  The non-const version is implemented below by inheriting from ltl::ConstIterator.
 *
 */
template<typename T, int N>
class MArrayIter : public MArrayIterConst<T,N>
{
   public:

      MArrayIter( MArray<T,N>& array )
            : MArrayIterConst<T,N>( array )
      { }

      MArrayIter( MArray<T,N>& array, const _iter_end_tag& E )
            : MArrayIterConst<T,N>( array, E )
      { }

      MArrayIter( const MArrayIter<T,N>& other )
            : MArrayIterConst<T,N>( other )
      { }

      //! Move the iterator to the next object.
      MArrayIter<T,N>& operator++()
      {
         MArrayIterConst<T,N>::operator++();
         return *this;
      }

      //! Dereference the iterator object. Return the element pointed to.
      T& operator*()
      {
         LTL_ASSERT( !(this->done()), "Dereferencing end iterator" );
         return *(this->data_);
      }
};

}

#endif // __LTL_ITER__
