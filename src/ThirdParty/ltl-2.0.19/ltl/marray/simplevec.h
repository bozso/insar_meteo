/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: simplevec.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_SMALLVEC__
#define __LTL_SMALLVEC__

#include <ltl/config.h>

namespace ltl {

//! Vector with known length at compile time, 1-based, STL-compatible.
/*! This simple class in its <tt>T = int</tt> instantiation
  is used to represent the index or indices of a particular element
  of an ltl::MArray.
  The second template parameter, namely the length \c N of the
  vector then is equal to the rank of the ltl::MArray this index
  refers to.
  This class is used to hold lists of indices for referencing
  arbitrary sets of elements of an ltl::MArray, e.g. the list of
  elements of a matrix which are \c ==0.
*/
template<class T, int N>
class FixedVector
{
   public:
      // STL-compatible type definitions
      typedef T              value_type;
      typedef T*             iterator;
      typedef const T*       const_iterator;
      typedef T&             reference;
      typedef const T&       const_reference;
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef std::reverse_iterator<iterator> reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

      static size_type size()
      {
         return N;
      }
      static bool empty()
      {
         return false;
      }
      static size_type max_size()
      {
         return N;
      }
      enum { static_size = N };

      // here we go

      FixedVector()
      { }

      FixedVector( const FixedVector<T,N>& other )
      {
         for( int i=1; i<=N; ++i )
            data_[i] = other.data_[i];
      }

      FixedVector<T,N>& operator=( const FixedVector<T,N>& o )
      {
         for( int i=1; i<=N; ++i )
            data_[i] = o.data_[i];
         return *this;
      }

      FixedVector<T,N>& operator=( const T& v )
      {
         for( int i=1; i<=N; ++i )
            data_[i] = v;
         return *this;
      }

      int length() const
      {
         return N;
      }

      int minIndex() const
      {
         return 1;
      }

      int maxIndex() const
      {
         return N;
      }

      //! Access the i-th element. Indexing is 1-based.
      T  operator()( int i ) const
      {
         return data_[i];
      }

      //! Access the i-th element. Indexing is 1-based.
      T& operator()( int i )
      {
         return data_[i];
      }

      T  operator[]( int i ) const
      {
         return data_[i];
      }

      T& operator[]( int i )
      {
         return data_[i];
      }

      //! Direct access to data.
      const T* data() const
      {
         return data_+1;
      }

      //! Return begin iterator.
      iterator begin()
      {
         return data_+1;
      }
      //! Return begin const_iterator.
      const_iterator begin() const
      {
         return data_+1;
      }

      //! Return end iterator.
      iterator end()
      {
         return data_+N+2;
      }
      //! Return end const_iterator.
      const_iterator end() const
      {
         return data_+N+2;
      }

      reverse_iterator rbegin()
      {
         return reverse_iterator(end());
      }
      const_reverse_iterator rbegin() const
      {
         return const_reverse_iterator(end());
      }

      reverse_iterator rend()
      {
         return reverse_iterator(begin());
      }
      const_reverse_iterator rend() const
      {
         return const_reverse_iterator(begin());
      }


   protected:
      T data_[N+1];
};


template<class T, int N>
std::ostream& operator<<(std::ostream& os, const FixedVector<T,N>& x)
{
   os << "FixedVector< T,"<<N<<" >" << std::endl;
   os << " [ ";
   for (int i=x.minIndex(); i <= x.maxIndex(); ++i)
   {
      os << x(i) << " ";
      if (!((i+1-x.minIndex())%9))
         os << std::endl << "  ";
   }
   os << " ]";
   return os;
}


}

#endif //__LTL_SMALLVEC__
