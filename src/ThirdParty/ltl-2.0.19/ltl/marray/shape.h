/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: shape.h 541 2014-07-09 17:01:12Z drory $
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
#error "<ltl/marray/shape.h> must be included via <ltl/marray.h>, never alone!"
#endif

#ifndef __LTL_SHAPE__
#define __LTL_SHAPE__

#include <ltl/config.h>

using std::istream;
using std::ostream;

namespace ltl {

class Range;

template <class T, int D> class MArray;

template<int N> class Shape;

template<int N> istream& operator>>( istream& is, Shape<N>& s );


template<int N>
class Shape
{
      template<class T, int D>
      friend class MArray;
      template<int D>
      friend class Shape;
      friend istream& operator>> <> ( istream& is, Shape<N>& s );

   public:
      Shape() { }
      Shape( const Range& r1 );
      Shape( const Range& r1, const Range& r2 );
      Shape( const Range& r1, const Range& r2, const Range& r3 );
      Shape( const Range& r1, const Range& r2, const Range& r3,
             const Range& r4 );
      Shape( const Range& r1, const Range& r2, const Range& r3,
             const Range& r4, const Range& r5 );
      Shape( const Range& r1, const Range& r2, const Range& r3,
             const Range& r4, const Range& r5, const Range& r6 );
      Shape( const Range& r1, const Range& r2, const Range& r3,
             const Range& r4, const Range& r5, const Range& r6,
             const Range& r7 );
      Shape( const int r1 );
      Shape( const int r1, const int r2 );
      Shape( const int r1, const int r2, const int r3 );
      Shape( const int r1, const int r2, const int r3, const int r4 );
      Shape( const int r1, const int r2, const int r3, const int r4, const int r5);
      Shape( const int r1, const int r2, const int r3, const int r4, const int r5,
             const int r6 );
      Shape( const int r1, const int r2, const int r3, const int r4, const int r5,
             const int r6, const int r7 );

      Shape( const Shape<N> &other  )
      {
         copy( other );
      }

      Shape& operator=( const Shape<N> &other  )
      {
         copy( other );
         return *this;
      }

      int  base( const int i ) const
      {
         return base_[i-1];
      }

      int& base( const int i )
      {
         return base_[i-1];
      }

      int  last( const int i ) const
      {
         return base_[i-1] + length_[i-1] - 1;
      }

      int  length( const int i ) const
      {
         return length_[i-1];
      }

      int& length( const int i )
      {
         return length_[i-1];
      }

      int  stride( const int i ) const
      {
         return stride_[i-1];
      }

      int& stride( const int i )
      {
         return stride_[i-1];
      }

      int  zeroOffset() const
      {
         return zeroOffset_;
      }

      int& zeroOffset()
      {
         return zeroOffset_;
      }

      bool isStorageContiguous( void ) const
      {
         return N==1 || isContiguous_;
      }

      bool isUnitStride( void ) const
      {
         return stride_[0]==1;
      }

      // defined as having equal lengths
      bool isConformable( const Shape& other ) const
      {
         for( int i=0; i<N; ++i )
            if( length_[i] != other.length_[i] )
               return false;
         return true;
      }

      int nelements() const
      {
         return nelements_;
      }

      // get a \c Shape object with dimension \c dim removed, used for partial reductions
      Shape<N-1> getShapeForContraction( const int dim ) const;

   protected:

      void calcIsStorageContiguous();
      void calcNelements();
      void calcZeroOffset();
      void copy( const Shape<N> &other  );
      void setupSelf( const int n );

      int  zeroOffset_; // offset from data_ to memory block
      int  base_[N];    // vector of bases
      int  length_[N];  // vector of lengths along each dim
      int  stride_[N];  // vector of strides

      bool isContiguous_;
      int  nelements_;
};


// --------------------------------------------------------------------
// CONSTRUCTORS
// --------------------------------------------------------------------


template<int N>
inline Shape<N>::Shape( const Range& r1 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   setupSelf(1);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   setupSelf(2);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2,
                        const Range& r3 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   length_[2] = r3.length();
   base_[2]   = r3.first();
   setupSelf(3);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2,
                        const Range& r3, const Range& r4 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   length_[2] = r3.length();
   base_[2]   = r3.first();
   length_[3] = r4.length();
   base_[3]   = r4.first();
   setupSelf(4);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2,
                        const Range& r3, const Range& r4,
                        const Range& r5 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   length_[2] = r3.length();
   base_[2]   = r3.first();
   length_[3] = r4.length();
   base_[3]   = r4.first();
   length_[4] = r5.length();
   base_[4]   = r5.first();
   setupSelf(5);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2,
                        const Range& r3, const Range& r4,
                        const Range& r5, const Range& r6 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   length_[2] = r3.length();
   base_[2]   = r3.first();
   length_[3] = r4.length();
   base_[3]   = r4.first();
   length_[4] = r5.length();
   base_[4]   = r5.first();
   length_[5] = r6.length();
   base_[5]   = r6.first();
   setupSelf(6);
}

template<int N>
inline Shape<N>::Shape( const Range& r1, const Range& r2,
                        const Range& r3, const Range& r4,
                        const Range& r5, const Range& r6,
                        const Range& r7 )
{
   length_[0] = r1.length();
   base_[0]   = r1.first();
   length_[1] = r2.length();
   base_[1]   = r2.first();
   length_[2] = r3.length();
   base_[2]   = r3.first();
   length_[3] = r4.length();
   base_[3]   = r4.first();
   length_[4] = r5.length();
   base_[4]   = r5.first();
   length_[5] = r6.length();
   base_[5]   = r6.first();
   length_[6] = r7.length();
   base_[6]   = r7.first();
   setupSelf(7);
}

// integer arguments
// base defaults to 1 !!!!!!
template<int N>
inline Shape<N>::Shape( const int r1 )
{
   LTL_ASSERT( r1>0, "Bad dimension length :"<<r1 );
   length_[0] = r1;
   base_[0]   = 1;
   setupSelf(1);
}

template<int N>
inline Shape<N>::Shape( const int r1, const int r2 )
{
   LTL_ASSERT( r1>0&&r2>0, "Bad dimension lengths :"<<r1<<","<<r2 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   setupSelf(2);
}

template<int N>
inline Shape<N>::Shape( const int r1, const int r2, const int r3 )
{
   LTL_ASSERT( r1>0&&r2>0&&r3>0, "Bad dimension lengths :"<<r1<<","<<r2<<","<<r3 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   length_[2] = r3;
   base_[2]   = 1;
   setupSelf(3);
}


template<int N>
inline Shape<N>::Shape( const int r1, const int r2,
                        const int r3, const int r4 )
{
   LTL_ASSERT( r1>0&&r2>0&&r3>0&&r4>0,
               "Bad dimension lengths :"<<r1<<","<<r2<<","<<r3<<","<<r4 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   length_[2] = r3;
   base_[2]   = 1;
   length_[3] = r4;
   base_[3]   = 1;
   setupSelf(4);
}


template<int N>
inline Shape<N>::Shape( const int r1, const int r2,
                        const int r3, const int r4, const int r5 )
{
   LTL_ASSERT( r1>0&&r2>0&&r3>0&&r4>0,
               "Bad dimension lengths :"<<r1<<","<<r2<<","<<r3<<","<<r4<<","<<r5 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   length_[2] = r3;
   base_[2]   = 1;
   length_[3] = r4;
   base_[3]   = 1;
   length_[4] = r5;
   base_[4]   = 1;
   setupSelf(5);
}

template<int N>
inline Shape<N>::Shape( const int r1, const int r2,
                        const int r3, const int r4, const int r5,
                        const int r6 )
{
   LTL_ASSERT( r1>0&&r2>0&&r3>0&&r4>0,
               "Bad dimension lengths :"<<r1<<","<<r2<<","<<r3<<","<<r4<<","<<r5<<","<<r6 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   length_[2] = r3;
   base_[2]   = 1;
   length_[3] = r4;
   base_[3]   = 1;
   length_[4] = r5;
   base_[4]   = 1;
   length_[5] = r6;
   base_[5]   = 1;
   setupSelf(6);
}

template<int N>
inline Shape<N>::Shape( const int r1, const int r2,
                        const int r3, const int r4, const int r5,
                        const int r6, const int r7 )
{
   LTL_ASSERT( r1>0&&r2>0&&r3>0&&r4>0,
               "Bad dimension lengths :"<<r1<<","<<r2<<","<<r3<<","<<r4<<","<<r5<<","<<r6<<","<<r7 );
   length_[0] = r1;
   base_[0]   = 1;
   length_[1] = r2;
   base_[1]   = 1;
   length_[2] = r3;
   base_[2]   = 1;
   length_[3] = r4;
   base_[3]   = 1;
   length_[4] = r5;
   base_[4]   = 1;
   length_[5] = r6;
   base_[5]   = 1;
   length_[6] = r7;
   base_[6]   = 1;
   setupSelf(7);
}



/*!
 * The storage is contiguous if for every dimension i = 1..N-1
 * stride(i)*length(i) == stride(i+1)
 * This assumes the dimensions are ordered! (c.f. blitz-version below which
 * does not require this assumption)
 */
template <int N>
inline void Shape<N>::calcIsStorageContiguous( void )
{
   isContiguous_ = true;
   for( int i=1; i<N; i++ )
      if( length(i)*stride(i) != stride(i+1) )
         isContiguous_ = false;
}


/*
 * The storage is contiguous if for the set
 * { | stride[i] * length[i] | }, i = 0..N_rank-1,
 * there is only one value which is not in the set
 * of strides; and if there is one stride which is 1.
 *
 * No assumption about the order of the dimensions is made!
 */
/*
template <int N>
void Shape<N>::calcIsStorageContiguous( void )
{
   int numStridesMissing = 0;
   int haveUnitStride = 0;
   int stride, vi, i, j;

   for( i=0; i<N; i++ )
   {
      stride = stride_[i];
      if( stride == 1 )
	 haveUnitStride = 1;

      vi = stride * length_[i];

      for( j=0; j<N; j++ )
	 if( stride_[j] == vi )
	    break;

      if( j == N )
      {
	 numStridesMissing++;
	 if( numStridesMissing == 2 )
	 {
	    isContiguous_= 0;
	    return;
	 }
      }
   }
   isContiguous_= haveUnitStride;
}
*/


template <int N>
inline void Shape<N>::calcNelements()
{
   nelements_=1;
   for( int i=0; i<N; i++ )
      nelements_ *= length_[i];
}


template <int N>
inline void Shape<N>::calcZeroOffset()
{
   zeroOffset_ = 0;
   for ( int i=0; i<N; i++ )
      zeroOffset_ -= stride_[i] * base_[i];
}


template <int N>
inline void Shape<N>::setupSelf( const int n )
{
   // check to see if the right constructor was used ...
   LTL_ASSERT( N>0, "Shape with <=0 dimensions constructed!" );
   #ifndef LTL_RANGE_CHECKING
   LTL_UNUSED(n);
   #endif
   LTL_ASSERT( n==N,
               "Wrong number of arguments in Shape<"<<N<<"> constructor!\n;"
               << "Got "<<n<<" Range arguments ...\n" );

   if( N==1 )
   {
      // specialisation for N=1 to make it easier for the compiler
      // to perform constant folding
      nelements_ = length_[0];
      zeroOffset_ = - base_[0];
      stride_[0] = 1;
      isContiguous_ = true;
   }
   else
   {
      nelements_=1;
      zeroOffset_ = 0;
      int s = 1;
      for( int i=0; i<N; i++ )
      {
         nelements_ *= length_[i];
         stride_[i] = s;
         zeroOffset_ -= s * base_[i];
         s *= length_[i];
      }
      calcIsStorageContiguous();
   }
}


template <int N>
void Shape<N>::copy( const Shape<N> &other )
{
   for( int i=1; i<=N; i++ )
   {
      base(i)   = other.base(i);
      length(i) = other.length(i);
      stride(i) = other.stride(i);
   }
   zeroOffset_ = other.zeroOffset();
   isContiguous_ = other.isStorageContiguous();
   nelements_ = other.nelements();
}

template <int N>
Shape<N-1> Shape<N>::getShapeForContraction( const int dim ) const
{
   Shape<N-1> s;
   for( int i=1, j=1; i<=N; ++i )
   {
      if( i == dim )
         continue;
      s.base(j)   = base(i);
      s.length(j) = length(i);
      s.stride(j) = stride(i);
      ++j;
   }
   s.zeroOffset_ = zeroOffset();
   s.isContiguous_ = isStorageContiguous();
   s.nelements_ = nelements();

   return s;
}



template<int N>
ostream& operator<<( ostream& os, const Shape<N>& s )
{
   os << "( ";
   for( int i=1; i<=N; i++ )
   {
      os << s.length(i);
      if( i!= N )
         os << " x ";
   }
   os << " ) : ";

   for( int i=1; i<=N; i++ )
      os << "(" << s.base(i)<< ","<< s.last(i)<<") ";
   return os;
}


template<int N>
istream& operator>>( istream& is, Shape<N>& s )
{
   int b,l;
   char c = ' ';
   while( c != ':' && !is.eof() )
      is >> c;
   if( c != ':' )
      throw IOException( "Expected ':' while reading MArray header!" );

   // now read (minIndex,maxIndex) pairs:

   for( int i=0; i<N; i++ )
   {
      is >> c;
      if( c != '(' || is.bad() )
         throw IOException( "Expected '(' while reading MArray header!" );

      is >> b;

      is >> c;
      if( c != ',' || is.bad() )
         throw IOException( "Expected ',' while reading MArray header!" );

      is >> l;

      s.base_[i] = b;
      s.length_[i] = l-b+1;

      is >> c;
      if( c != ')' || is.bad() )
         throw IOException( "Expected ')' while reading MArray header!" );
   }

   s.setupSelf( N );
   return is;
}

}

#endif // __LTL_SHAPE__
