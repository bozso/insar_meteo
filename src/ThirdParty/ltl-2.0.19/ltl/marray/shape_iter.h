/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: shape_iter.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __LTL_MARRAY__
#error "<ltl/marray/shape_iter.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_SHAPE_ITER__
#define __LTL_SHAPE_ITER__

#include <ltl/config.h>

namespace ltl {

//! A little helper class for the evaluate_with index implementation.
//
template<int N>
class ShapeIter
{

   public:
      // this is the iterator positioned at end()
      ShapeIter()
            : stride_(1), shape_(NULL), done_(true)
      { }

      ShapeIter( const Shape<N>& shape );

      ShapeIter( const ShapeIter<N>& other )
            : data_(other.data_), first_(other.first_), stride_(other.stride_),
            shape_(other.shape_), done_(other.done_)
      {
         //cout << "copy() \n";
         for( int i=0; i<N; ++i )
         {
            length_[i]  = other.length_[i];
            strides_[i] = other.strides_[i];
            stack_[i]   = other.stack_[i];
            last_[i]    = other.last_[i];
         }
      }

      // increment the iterator
      // this method is almost never used, see comment below...
      inline ShapeIter<N>& operator++()
      {
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

      ShapeIter<N> operator++( int )
      {
         ShapeIter<N> tmp (*this);
         ++(*this);
         return tmp;
      }

      // provide separate versions of advance() for incrementing the innermost
      // loop and needAdvanceDim()/advanceDim() for incrementing an outer loop
      // since when evaluating an expression involving more than one iterator
      // all terms MUST have the same geometry, so the loop structure is
      // identical and it's sufficent to check the end of loop condition on
      // one of the iterators ... all others are then 'remote controlled'
      // via the following methods.
      inline void advance()
      {
         data_ += stride_;
      }

      inline void advanceWithStride1()
      {
         data_ ++;
      }

      inline bool needAdvanceDim() const
      {
         if( N == 1 )
            return false;
         else
            return data_ == last_[0];
      }

      void advanceDim();

      // for 1-d arrays, we can avoid needAdvanceDim() since it's
      // equivalent to the done condition
      inline bool done() const
      {
         if( N == 1 )  // if() will be const-folded away ...
            return data_ == last_[0];
         else
            return done_;
      }

      bool isStride1() const
      {
         return stride_==1;
      }

      //! inquire about the length of the loop over each dimension
      int length( const int i ) const
      {
         return length_[i-1];
      }

      const Shape<N>* shape() const
      {
         return shape_;
      }

   protected:
      int data_;
      int first_;
      const int stride_;
      const Shape<N> *shape_;
      bool done_;

      int stack_[N], last_[N];
      int strides_[N], length_[N];
};


//! The constructor implementation.
//
template<int N>
ShapeIter<N>::ShapeIter( const Shape<N>& shape )
      : data_(shape.base(1)), first_(shape.base(1)),
      stride_(shape.stride(1)), shape_(&shape), done_(false)
{
   // copy information to avoid dereferencing array all the time
   for( int i=0; i<N; i++ )
   {
      strides_[i] = shape.stride(i+1);
      length_[i] = shape.length(i+1);
      stack_[i] = data_;
      last_[i] = data_ + length_[i] * strides_[i];
   }
}


//! Reset stacks after we've hit the end of a dimension.
//
template<int N>
void ShapeIter<N>::advanceDim()
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
      done_ = true;
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

}

#endif
