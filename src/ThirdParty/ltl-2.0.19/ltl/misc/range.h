/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: range.h 491 2011-09-02 19:36:39Z drory $
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




#ifndef __LTL_RANGE__
#define __LTL_RANGE__

#include <ltl/config.h>

#include <climits>   // INT_MIN

namespace ltl {

enum { minStart = INT_MIN,  minEnd = INT_MIN };

//! Class to enable referencing of ltl::MArray index ranges.
/*! 
  A ltl::Range is a utility object used in the context of
  ltl::MArray describing a range of integers used for creating arrays
  and indexing an array along one dimension.
  For example, the set of indices <tt>{-10, -8, ..., 6, 8, 10}</tt>
  is represented by the object <tt>Range( -10, 10, 2 )</tt>.
*/
class Range
{
   public:

      Range() :
            first_(minStart),
            last_(minEnd),
            stride_(1),
            length_(1)
      {}

      //! Construct a Range object. 
      /*! The stride arguments defaults to 1 and may be omitted.
        <tt>last \>= first</tt> is required.
      */
      Range( const int first, const int last, const int stride=1 ) :
            first_(first), last_(last),
            stride_(stride),
            length_((last-first)/stride+1)
      {
         LTL_ASSERT( (first==minStart)||(last>=first && length_>0),
                     "Bad Range("<<first<<","<<last<<")" );
      }

      Range( const Range& other ) :
            first_(other.first_),
            last_(other.last_),
            stride_(other.stride_),
            length_(other.length_)
      { }

      //! Return the first index.
      inline int first( int lowest=minStart ) const
      {
         if( first_ == minStart )
            return lowest;
         return first_;
      }

      //! Return the last index.
      inline int last( int highest=minEnd ) const
      {
         if( last_ == minEnd )
            return highest;
         return last_;
      }

      //! Return the number of indices in the range.
      inline int length() const
      {
         return length_;
      }

      //! Return the stride.
      inline int stride() const
      {
         return stride_;
      }

      //! This \em static member returns a special Range object that represents the entire dimension, wherever used.
      /*! This is useful to refer to an entire dimension
        without having to figure out the actual size.
       */
      static Range all()
      {
         return Range( minStart, minEnd, 1 );
      }

      //! Return a negative shifted Range object.
      /*! I.e. one whose lower and upper indices
        are lower by a number \c shift.
       */
      Range operator-( const int shift ) const
      {
         return Range( first_ - shift, last_ - shift, stride_ );
      }

      //! Shift the Range object by \c shift to the negative.
      /*! I.e. the lower and upper indices
        are lowered by \c shift.
       */
      Range operator-=( const int shift )
      {
         first_ -= shift;
         last_ -= shift;
         return *this;
      }

      //! Return a positive shifted Range object.
      /*! I.e. one whose lower and upper indices
        are higher by a number \c shift.
      */
      Range operator+( const int shift ) const
      {
         return Range(first_ + shift, last_ + shift, stride_);
      }

      //! Shift the Range object by \c shift to the positive.
      /*! I.e. the lower and upper indices
        are increased by \c shift.
       */
      Range operator+=( const int shift )
      {
         first_ += shift;
         last_ += shift;
         return *this;
      }

      //! Copy a Range object
      Range operator=( const Range &other )
      {
         first_ = other.first_;
         last_ = other.last_;
         stride_ = other.stride_;
         length_ = other.length_;
         return *this;
      }

   private:
      int first_;
      int last_;
      int stride_;
      int length_;
};

}

#endif // __LTL_RANGE__

