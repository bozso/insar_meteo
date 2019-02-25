/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvmemory.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FVECTOR__
#error "<ltl/fvector/fvmemory.h> must be included via <ltl/fvector.h>, never alone!"
#endif


#ifndef __LTL_FV_MEMORY__
#define __LTL_FV_MEMORY__

#include <ltl/config.h>

namespace ltl {

//! Memory and data access interface for ltl::FVMemory.

/*! \ingroup fvector_fmatrix
 *
 *  The ltl::FVMemory objects implement the memory access interface of
 *  the ltl::FVector object, i.e. operator() for 1-based access and
 *  operator[] for 0-based access. References \c N objects of type \c T accessible
 *  through a \c T* with stride \c S.
 *
 *  The template parameter S is the stride which may be != 1, such that
 *  column vectors of ltl::FVMatrix can be referenced as ltl::FVector objects.
 *
 *  There is a general implementation (this one) and a specialization for S=0.
 *  S!=0 indicates that this instance is a reference to foreign storage. S is then
 *  the stride used to access the memory elements. The specialization for S=0
 *  below is used for instances that hold there own memory (i.e. ltl::FVector objects)
 *  that have their own memory, not references to other ltl::FVector objects. In this
 *  case the physical stride is one.
 *
 *  \c FVMemory does not require more memory than is necessary
 *  to store the N elements of the vector if S=0. If S>0, the size is exactly
 *  \c sizeof(T*).
 *
 */
template<class T, int N, int S>
class FVMemory
{
   public:
      inline FVMemory()
      { }

      //! construct as a reference to the memory pointed to by \c data.
      inline FVMemory(  T* const data )
         : __data_(data)
      { }

      //@{
      //! Access elements with zero-based index \c i.
      inline T operator[]( const int i ) const
      {
         LTL_ASSERT( i>=0 && i<N,
                     "Direct index "<<i<<" out of bounds in FVector[] of length "<<N );
         return __data_[i*S];
      }

      //! Access elements with zero-based index \c i.
      inline T& operator[]( const int i )
      {
         LTL_ASSERT( i>=0 && i<N,
                     "Direct index "<<i<<" out of bounds in FVector[] of length "<<N );
         return __data_[i*S];
      }
      //@}

      //@{
      //! Access elements with one-based index \c i.
      inline T operator()( const int i ) const
      {
         LTL_ASSERT( i>0 && i<=N,
                     "Index "<<i<<" out of bounds in FVector() of length "<<N );
         return __data_[(i-1)*S];
      }

      //! Access elements with one-based index \c i.
      inline T& operator()( const int i )
      {
         LTL_ASSERT( i>0 && i<=N,
                     "Index "<<i<<" out of bounds in FVector() of length "<<N );
         return __data_[(i-1)*S];
      }
      //@}

      //! Return a pointer to the data.
      inline T* data()
      {
         return __data_;
      }

      //! Return a const pointer to the data.
      inline T* data() const
      {
         return __data_;
      }

#ifndef __SUNPRO_CC
   protected:
#endif
      T* __data_;  //!< points to first element of data block for 0-based access
};



/*!
 *  Specialization for \c FVMemory instances having their own memory
 *  embedded. Indicated by \c S=0.
 *  This is the case for every freshly allocated FVector object
 *  having its own memory.
 *
 *  The physical stride for memory access is one in this case.
 */
template<class T, int N>
class FVMemory<T,N,0>
{
   public:
      //! construct as a reference to the memory pointed to by \c data.
      inline FVMemory()
      { }

      //! copy data over from memory pointed to by \c data.
      inline FVMemory( const T* const restrict_ data )
      {
         for( int i=0; i<N; ++i )
            (*this)[i] = data[i];
      }

      //! copy constructor
      inline FVMemory( const FVMemory<T,N,0>& restrict_ other )
      {
         for( int i=0; i<N; ++i )
            (*this)[i] = other[i];
      }

      //@{
      //! Access elements with zero-based index \c i.
      inline T operator[]( const int i ) const
      {
         LTL_ASSERT( i>=0 && i<N,
                     "Direct index "<<i<<" out of bounds in FVector[] of length "<<N );
         return __data_[i];
      }

      //! Access elements with zero-based index \c i.
      inline T& operator[]( const int i )
      {
         LTL_ASSERT( i>=0 && i<N,
                     "Direct index "<<i<<" out of bounds in FVector[] of length "<<N );
         return __data_[i];
      }
      //@}

      //@{
      //! Access elements with one-based index \c i.
      inline T operator()( const int i ) const
      {
         LTL_ASSERT( i>0 && i<=N,
                     "Index "<<i<<" out of bounds in FVector() of length "<<N );
         return __data_[i-1];
      }

      //! Access elements with one-based index \c i.
      inline T& operator()( const int i )
      {
         LTL_ASSERT( i>0 && i<=N,
                     "Index "<<i<<" out of bounds in FVector() of length "<<N );
         return __data_[i-1];
      }
      //@}

      //@{
      //! Return a pointer to the data.
      inline T* data()
      {
         return __data_;
      }

      //! Return a const-pointer to the data.
      inline const T* data() const
      {
         return __data_;
      }
      //@}

#ifndef __SUNPRO_CC
   protected:
#endif
      T  __data_[N];        //!< the memory block
};

}

#endif // __LTL_FV_MEMORY__
