/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fviter.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/fvector/fviter.h> must be included via <ltl/fvector.h>, never alone!"
#endif


#ifndef __LTL_FV_ITER__
#define __LTL_FV_ITER__

#include <ltl/config.h>

#include <iterator>

namespace ltl {

/*! \file fviter.h
  Iterators for ltl::FVector objects. These work for both 'real' ltl::FVectors 
  and ltl::FVectors referencing someone else's memory.
*/

struct fviter_end_tag
{ };
   

template<class T, int N, int S>
class FVIterConst
{
   public:
      typedef std::bidirectional_iterator_tag          iterator_category;
      typedef typename FVector<T,N,S>::value_type      value_type;
      typedef typename FVector<T,N,S>::const_reference const_reference;
      typedef typename FVector<T,N,S>::reference       reference;
      typedef typename FVector<T,N,S>::pointer         pointer;
      typedef typename FVector<T,N,S>::const_pointer   const_pointer;
      typedef typename FVector<T,N,S>::difference_type difference_type;

      enum { static_size = 1 };         

      FVIterConst( const FVector<T,N,S>& v )
         : data_( const_cast<T*>(v.__data_) )
      { }
      
      FVIterConst( const FVector<T,N,S>& v, fviter_end_tag )
         : data_( const_cast<T*>(v.__data_ + v.length()*S ) )
      { }
      
      // Note: zero-based access
      T operator[]( const int i ) const
      {
         return data_[i*S];
      }

      T operator*() const
      {
         return *data_;
      }
      
      FVIterConst<T,N,S>& operator++()
      {
         data_ += S;
         return *this;
      }
      
      FVIterConst<T,N,S> operator++( int )
      {
         FVIterConst<T,N,S> tmp( *this );
         data_ += S;
         return tmp;
      }
      
      FVIterConst<T,N,S>& operator--()
      {
         data_ -= S;
         return *this;
      }
      
      FVIterConst<T,N,S> operator--( int )
      {
         FVIterConst<T,N,S> tmp( *this );
         data_ -= S;
         return tmp;
      }
      
      template<int P>
      bool operator==( const FVIterConst<T,N,P>& other ) const
      {
         return data_ == other.data_;
      }

      template<int P>
      bool operator!=( const FVIterConst<T,N,P>& other ) const
      {
         return data_ != other.data_;
      }

   protected:
      T* restrict_ data_;
};


template<class T, int N>
class FVIterConst<T,N,0>
{
   public:
      typedef std::bidirectional_iterator_tag          iterator_category;
      typedef typename FVector<T,N,0>::value_type      value_type;
      typedef typename FVector<T,N,0>::const_reference const_reference;
      typedef typename FVector<T,N,0>::reference       reference;
      typedef typename FVector<T,N,0>::pointer         pointer;
      typedef typename FVector<T,N,0>::const_pointer   const_pointer;
      typedef typename FVector<T,N,0>::difference_type difference_type;

      enum { static_size = 1 };         

      FVIterConst( const FVector<T,N,0>& v )
         : data_( const_cast<T*>(v.__data_) )
      { }
      
      FVIterConst( const FVector<T,N,0>& v, fviter_end_tag )
         : data_( const_cast<T*>(v.__data_ + v.length()) )
      { }
      
      // Note: zero-based access
      T operator[]( const int i ) const
      {
         return data_[i];
      }

      T operator*() const
      {
         return *data_;
      }
      
      FVIterConst<T,N,0>& operator++()
      {
         ++data_;
         return *this;
      }
      
      FVIterConst<T,N,0> operator++( int )
      {
         FVIterConst<T,N,0> tmp( *this );
         ++data_;
         return tmp;
      }
      
      FVIterConst<T,N,0>& operator--()
      {
         --data_;
         return *this;
      }
      
      FVIterConst<T,N,0> operator--( int )
      {
         FVIterConst<T,N,0> tmp( *this );
         --data_;
         return tmp;
      }
      
      template<int S>
      bool operator==( const FVIterConst<T,N,S>& other ) const
      {
         return data_ == other.data_;
      }

      template<int S>
      bool operator!=( const FVIterConst<T,N,S>& other ) const
      {
         return data_ != other.data_;
      }


   protected:
      T* restrict_ data_;
};


template<class T, int N, int S>
class FVIter
{
   public:
      typedef std::forward_iterator_tag                iterator_category;
      typedef typename FVector<T,N,S>::value_type      value_type;
      typedef typename FVector<T,N,S>::const_reference const_reference;
      typedef typename FVector<T,N,S>::reference       reference;
      typedef typename FVector<T,N,S>::pointer         pointer;
      typedef typename FVector<T,N,S>::const_pointer   const_pointer;
      typedef typename FVector<T,N,S>::difference_type difference_type;

       enum { static_size = 1 };         

      FVIter( FVector<T,N,S>& v )
         : data_( v.__data_ )
      { }
      
      FVIter( FVector<T,N,S>& v, fviter_end_tag )
         : data_( v.__data_ + v.length()*S )
      { }
      
      T operator[]( const int i ) const
      {
         return data_[i*S];
      }

      T& restrict_ operator[]( const int i )
      {
         return data_[i*S];
      }

      T operator*() const
      {
         return *data_;
      }
      
      T& restrict_ operator*()
      {
         return *data_;
      }
      
      FVIter<T,N,S>& operator++()
      {
         data_ += S;
         return *this;
      }

      FVIter<T,N,S> operator++( int )
      {
         FVIter<T,N,S> tmp( *this );
         data_ += S;
         return tmp;
      }
      
      FVIter<T,N,S>& operator--()
      {
         data_ -= S;
         return *this;
      }
      
      FVIter<T,N,S> operator--( int )
      {
         FVIter<T,N,S> tmp( *this );
         data_ -= S;
         return tmp;
      }

      template<int P>
      bool operator==( const FVIter<T,N,S>& other ) const
      {
         return data_ == other.data_;
      }

      template<int P>
      bool operator!=( const FVIter<T,N,S>& other ) const
      {
         return data_ != other.data_;
      }

   protected:
      T* restrict_ data_;
};


template<class T, int N>
class FVIter<T,N,0>
{
   public:
      typedef std::forward_iterator_tag                iterator_category;
      typedef typename FVector<T,N,0>::value_type      value_type;
      typedef typename FVector<T,N,0>::const_reference const_reference;
      typedef typename FVector<T,N,0>::reference       reference;
      typedef typename FVector<T,N,0>::pointer         pointer;
      typedef typename FVector<T,N,0>::const_pointer   const_pointer;
      typedef typename FVector<T,N,0>::difference_type difference_type;

       enum { static_size = 1 };         

      FVIter( FVector<T,N,0>& v )
         : data_( v.__data_ )
      { }
      
      FVIter( FVector<T,N,0>& v, fviter_end_tag )
         : data_( v.__data_ + v.length() )
      { }
      
      T operator[]( const int i ) const
      {
         return data_[i];
      }

      T& restrict_ operator[]( const int i )
      {
         return data_[i];
      }

      T operator*() const
      {
         return *data_;
      }
      
      T& restrict_ operator*()
      {
         return *data_;
      }
      
      FVIter<T,N,0>& operator++()
      {
         ++data_;
         return *this;
      }

      FVIter<T,N,0> operator++( int )
      {
         FVIter<T,N,0> tmp( *this );
         ++data_;
         return tmp;
      }
      
      FVIter<T,N,0>& operator--()
      {
         --data_;
         return *this;
      }
      
      FVIter<T,N,0> operator--( int )
      {
         FVIter<T,N,0> tmp( *this );
         --data_;
         return tmp;
      }
      
      template<int S>
      bool operator==( const FVIter<T,N,S>& other ) const
      {
         return data_ == other.data_;
      }

      template<int S>
      bool operator!=( const FVIter<T,N,S>& other ) const
      {
         return data_ != other.data_;
      }


   protected:
      T* restrict_ data_;
};

}

#endif // __LTL_FV_ITER__
