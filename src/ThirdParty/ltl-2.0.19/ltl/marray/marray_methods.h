/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marray_methods.h 557 2015-03-10 18:37:47Z drory $
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

#ifndef __LTL_IN_FILE_MARRAY__
#error "<ltl/marray/marray_methods.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_MARRAYMETHODS__
#define __LTL_MARRAYMETHODS__

#include <ltl/config.h>

namespace ltl {

// --------------------------------------------------------------------
// PUBLIC METHODS
// --------------------------------------------------------------------

// Construct from preexisting data.
//
template <typename T, int N>
MArray<T,N>::MArray( T *data, const int * dims )
{
   setupShape( dims );
   memBlock_ = new MemoryBlock<T>( data );
   data_ = memBlock_->data();
   data_ += zeroOffset();
}

template <typename T, int N>
MArray<T,N>::MArray( const string filename, const int * dims )
{
   setupShape( dims );
   setupMemory(true, filename.c_str());
}

// Construct from expression.
template <typename T, int N>
template<typename Expr>
MArray<T,N>::MArray( const ExprNode<Expr,N>& e, const bool map, const char * filename )
{
   shape_ = *e.shape();
   shape_.setupSelf(N);

   setupMemory(map, filename);

   eval_assign_expr( *this, const_cast<ExprNode<Expr,N>&>(e) );
}

// Construct from shape.
//
template<typename T, int N>
MArray<T,N>::MArray( const Shape<N>* s, const bool map, const char * filename )
{
   shape_ = *s;
   shape_.setupSelf(N);
   setupMemory(map, filename);
}

// Makes this being a referece to other's data.
//
template<typename T, int N>
void MArray<T,N>::makeReference( const MArray<T,N>& other )
{
   if( !empty() )
      memBlock_->removeReference();

   shape_      = other.shape_;
   memBlock_   = other.memBlock_;
   if( memBlock_ )
   {
      // increment reference count of memblock
      memBlock_->addReference();
      // get data pointer (maybe altered due to e.g. slice etc.)
      data_       = other.data_;
   }
}

// Make a reference as a different-dimensional view of another \c MArray's data
//
template<typename T, int N>
template<int N2>
void MArray<T,N>::makeReferenceWithDims( const MArray<T,N2>& other, const int* dims )
{
   LTL_ASSERT (other.isStorageContiguous(), "Attempt to reshape array that whose storage is not contiguous");
   if( !empty() )
      memBlock_->removeReference();

   setupShape( dims );
   LTL_ASSERT (other.nelements() == nelements(), "Given dimensions do not have the same number of elements as source array in reshape");

   memBlock_   = other.memoryBlock();
   if( memBlock_ )
   {
      // increment reference count of memblock
      data_ = memBlock_->data();
      data_  += shape_.zeroOffset();  // data points to point (0,0,...,0)
   }
}




// Reallocate memory. Data are abolished.
//
template<typename T, int N>
void MArray<T,N>::realloc( const Shape<N>& s,
                           const bool map, const char * filename )
{
   if( !empty() )
      memBlock_->removeReference();
   shape_ = s;
   shape_.setupSelf(N);
   setupMemory(map, filename);
}


// Copy from an expression.
//
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator=( const ExprNode<Expr,N>& e )
{
   if( empty() )
   {
      shape_ = *e.shape();
      shape_.setupSelf(N);
      setupMemory(false, NULL);
   }
   eval_assign_expr( *this, const_cast<ExprNode<Expr,N>&>(e) );
   return *this;
}



template <typename T, int N>
void MArray<T,N>::setBase( const int b1 )
{
   ASSERT_DIM(1);
   int offset = (shape_.base(1)-b1) * shape_.stride(1);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1, const int b2 )
{
   ASSERT_DIM(2);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1,
                           const int b2, const int b3 )
{
   ASSERT_DIM(3);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2) +
                (shape_.base(3)-b3) * shape_.stride(3);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
   shape_.base(3) = b3;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1, const int b2, const int b3,
                           const int b4 )
{
   ASSERT_DIM(4);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2) +
                (shape_.base(3)-b3) * shape_.stride(3) +
                (shape_.base(4)-b4) * shape_.stride(4);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
   shape_.base(3) = b3;
   shape_.base(4) = b4;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1, const int b2, const int b3,
                           const int b4, const int b5 )
{
   ASSERT_DIM(5);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2) +
                (shape_.base(3)-b3) * shape_.stride(3) +
                (shape_.base(4)-b4) * shape_.stride(4) +
                (shape_.base(5)-b5) * shape_.stride(5);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
   shape_.base(3) = b3;
   shape_.base(4) = b4;
   shape_.base(5) = b5;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1, const int b2, const int b3,
                           const int b4, const int b5, const int b6 )
{
   ASSERT_DIM(6);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2) +
                (shape_.base(3)-b3) * shape_.stride(3) +
                (shape_.base(4)-b4) * shape_.stride(4) +
                (shape_.base(5)-b5) * shape_.stride(5) +
                (shape_.base(6)-b6) * shape_.stride(6);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
   shape_.base(3) = b3;
   shape_.base(4) = b4;
   shape_.base(5) = b5;
   shape_.base(6) = b6;
}

template <typename T, int N>
void MArray<T,N>::setBase( const int b1, const int b2, const int b3,
                           const int b4, const int b5, const int b6,
                           const int b7 )
{
   ASSERT_DIM(6);
   int offset = (shape_.base(1)-b1) * shape_.stride(1) +
                (shape_.base(2)-b2) * shape_.stride(2) +
                (shape_.base(3)-b3) * shape_.stride(3) +
                (shape_.base(4)-b4) * shape_.stride(4) +
                (shape_.base(5)-b5) * shape_.stride(5) +
                (shape_.base(6)-b6) * shape_.stride(6) +
                (shape_.base(7)-b7) * shape_.stride(7);
   shape_.zeroOffset() += offset;
   data_ += offset;
   shape_.base(1) = b1;
   shape_.base(2) = b2;
   shape_.base(3) = b3;
   shape_.base(4) = b4;
   shape_.base(5) = b5;
   shape_.base(6) = b6;
   shape_.base(7) = b7;
}

// --------------------------------------------------------------------
// reorder array without copy
// --------------------------------------------------------------------

/*! Reverse dimension dim, keep index range unchanged, such that the following
  holds:   A'( I ) = A( last-I+1 ). Note that the base is unchanged, so that
  if A(3) refered to the first element, now A'(3) holds the value of the
  former last element.
  Equation for offset: \n
  data_ + last_ * stride_ = data_ + offset + base * stride * (-1)
*/
template <typename T, int N>
void MArray<T, N>::reverseSelf( const int dim )
{
   CHECK_DIM(dim);

   const int offset = shape_.stride(dim) * (shape_.base(dim) + shape_.last(dim));
   data_ += offset;
   shape_.stride(dim) *= (-1);
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
}

template <typename T, int N>
MArray<T,N> MArray<T,N>::reverse( const int dim1 ) const
{
   MArray<T, N> A(*this);
   A.reverseSelf( dim1 );
   return A;
}


template <typename T, int N>
void MArray<T, N>::transposeSelf( const int dim1, const int dim2 )
{
   CHECK_DIM(dim1);
   CHECK_DIM(dim2);
   LTL_ASSERT(dim1 != dim2, "Trying to transpose same dimensions: " << dim1 << " " << dim2 );

   std::swap(shape_.length(dim1), shape_.length(dim2));
   std::swap(shape_.stride(dim1), shape_.stride(dim2));

   shape_.calcIsStorageContiguous();
}


template <typename T, int N>
MArray<T,N> MArray<T, N>::transpose( const int dim1, const int dim2 ) const
{
   MArray<T,N> A(*this);
   A.transposeSelf( dim1, dim2 );
   return A;
}


// --------------------------------------------------------------------
// DEBUG STUFF, OUTPUT
// --------------------------------------------------------------------


template <typename T, int N>
void MArray<T,N>::describeSelf() const
{
   cout << "  Num elements        : " << nelements() << endl;
   cout << "  Zero offset         : " << zeroOffset() << endl;
   cout << "  Data (0,0,...)      : " << data_ << endl;
   cout << "  First element       : " << data() << endl;
   cout << "  Storage Contiguous  : " << isStorageContiguous() << endl;
   if( !empty() )
      memBlock_->describeSelf();
   cout << "  Dimensions          : " << shape_ << endl;
   cout << "  Strides             : ";
   for( int i=1; i<=N; ++i )
      cout << stride(i) << " ";
   cout << endl;
}




// --------------------------------------------------------------------
// PRIVATE METHODS
// --------------------------------------------------------------------

template <typename T, int N>
template<typename T2>
inline void MArray<T,N>::copy( const MArray<T2,N>& other )
{
   // make sure that copying an uninitialized MArray yields
   // an uninitialized MArray
   if( other.empty() )
   {
      if( !empty() )
            memBlock_->removeReference();
      memBlock_ = NULL;
      return;
   }

   // if we are yet uninitialized, initialize
   if( empty() )
   {
      shape_ = *other.shape();
      shape_.setupSelf(N);
      setupMemory(false, NULL);
   }
   // finally, copy data
   ExprNode<typename MArray<T2,N>::ConstIterator,N> i( other.begin() );
   eval_assign_expr( *this, i );
}


template <typename T, int N>
inline void MArray<T,N>::fill( const T t )
{
   LTL_ASSERT( !empty(), "Assignment to uninitialized MArray from expression with unknown shape (scalar)." );
   ExprNode<ExprLiteralNode<T>,N> e(t);
   eval_assign_expr( *this, e );
}


template <typename T, int N>
void MArray<T,N>::setupMemory( const bool map, const char * filename )
{
   // Allocate a block of memory
   if(map)
      memBlock_ = new MemoryMapBlock<T>( shape_.nelements(), filename );
   else
      memBlock_ = new MemoryBlock<T>( shape_.nelements() );
   data_     = memBlock_->data();

   //shape_.calcZeroOffset();
   data_    += shape_.zeroOffset();  // data points to point (0,0,...,0)
}

template <typename T, int N>
void MArray<T,N>::setupShape( const int * dims )
{
   for( int i=0; i<N; i++ )
   {
      shape_.base_[i] = 1;
      shape_.length_[i] = dims[i];
   }
   shape_.setupSelf(N);
}


// Constructs a pure subarray of other, i.e. rank is preserved.
//
template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1 )
{
   makeReference( other );
   setrange( 1, r1 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2,
                            const Range& r3 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   setrange( 3, r3 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}

template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2,
                            const Range& r3, const Range& r4 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   setrange( 3, r3 );
   setrange( 4, r4 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2,
                            const Range& r3, const Range& r4,
                            const Range& r5 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   setrange( 3, r3 );
   setrange( 4, r4 );
   setrange( 5, r5 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2,
                            const Range& r3, const Range& r4,
                            const Range& r5, const Range& r6 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   setrange( 3, r3 );
   setrange( 4, r4 );
   setrange( 5, r5 );
   setrange( 6, r6 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


template <typename T, int N>
void MArray<T,N>::subarray( const MArray<T,N>& other,
                            const Range& r1, const Range& r2,
                            const Range& r3, const Range& r4,
                            const Range& r5, const Range& r6,
                            const Range& r7 )
{
   makeReference( other );
   setrange( 1, r1 );
   setrange( 2, r2 );
   setrange( 3, r3 );
   setrange( 4, r4 );
   setrange( 5, r5 );
   setrange( 6, r6 );
   setrange( 7, r7 );
   shape_.calcZeroOffset();
   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}


/*! 'Cut out' the ltl::Range r in dimension dim, used for rank-preserving
  subarrays. Range is reset to base of 1, such that if you have Range(3,10)
  and you cut out Range(5,8) you get a dimension having Range(1:3).

  Equation for offset: \n
  data_ + f*stride_(dim) = data_ + offset + 1*stride_(dim)*s
*/
template <typename T, int N>
void MArray<T,N>::setrange( const int dim, const Range& r )
{
   CHECK_RANGE(r,dim);

   const int f = r.first( minIndex(dim) );
   const int l = r.last( maxIndex(dim) );
   const int s = r.stride();

   // new base is 1 !!!
   shape_.length(dim) = (l - f) / s + 1;

   data_ += (f - s) *shape_.stride(dim);

   shape_.stride(dim) *= s;
   shape_.base(dim) = 1;
}

/*! Weird version of slicearray dealing with the case that some dimensions
  are slices, i.e. we have a mixture of Range and int arguments.
  There are two versions of slice() below being called for each dimension:
  one version has a Range argument, the other an int argument.
*/
template<typename T, int N>
template<int N2, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7>
void MArray<T,N>::slicearray( const MArray<T,N2>& other,
                              R1 r1, R2 r2, R3 r3, R4 r4, R5 r5, R6 r6, R7 r7 )
{
   // cannot call makeReference since we might be a slice and therefore
   // have different rank ...
   memBlock_           = other.memoryBlock();
   shape_.zeroOffset() = other.zeroOffset();
   // need this to increment reference count
   data_               = memBlock_->data();
   data_               = other.data_; // data might have changed by slicing

   int destDim = 1; // counter for destination dimensions

   slice( destDim, r1, other, 1 );
   slice( destDim, r2, other, 2 );
   slice( destDim, r3, other, 3 );
   slice( destDim, r4, other, 4 );
   slice( destDim, r5, other, 5 );
   slice( destDim, r6, other, 6 );
   slice( destDim, r7, other, 7 );

   shape_.calcZeroOffset();

   shape_.calcIsStorageContiguous();
   shape_.calcNelements();
}

//! ltl::Range version of slice(): Essentially a subarray in this dimension.
//
template<typename T, int N>
template<int N2>
void MArray<T,N>::slice( int& destDim, Range r,
                         const MArray<T,N2>& other,
                         int sourceDim )
{
   // copy shape parameters from sourceRank
   shape_.base(destDim)   = other.minIndex(sourceDim);
   shape_.length(destDim) = other.length(sourceDim);
   shape_.stride(destDim) = other.stride(sourceDim);

   setrange( destDim, r );

   ++destDim; // Range parameter : dimension preserved
}


//! \c int version of slice(): slice in this dimension, rank is reduced by one.
//
template<typename T, int N>
template<int N2>
inline void  MArray<T,N>::slice( int& /*destDim*/, int r,
                                 const MArray<T,N2>& other,
                                 int sourceDim )
{
   CHECK_SLICE(r,sourceDim);

   // simply slice dimension away:
   data_  += r * other.stride(sourceDim);
}

}

#endif
