/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmiter.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FMATRIX__
#error "<ltl/fmatrix/fmiter.h> must be included via <ltl/fmatrix.h>, never alone!"
#endif


#ifndef __LTL_FM_ITER__
#define __LTL_FM_ITER__

#include <ltl/config.h>

namespace ltl {

//
// Iterators for FVector objects. These work for both 'real' FVectors 
// and FVectors referencing someone else's memory
//

template<class T, int M, int N>
class FMIterConst
{
   public:
      enum { static_size = 1 };

      FMIterConst( const FMatrix<T,M,N>& m )
         : __data_( const_cast<T*>(m.__data_) )
      { }
      
      // this is 0-based! (operator[] cannot take two args)
      T operator[]( const int i ) const
      {
         return __data_[i];
      }

      T operator()( const int i, const int j ) const
      {
         return __data_[(i-1)*N+(j-1)];
      }

   protected:
      T* restrict_  __data_;
};


template<class T, int M, int N>
class FMIter
{
   public:
      enum { static_size = 1 };

      FMIter( FMatrix<T,M,N>& m )
         : __data_( m.__data_ )
      { }
      
      // this is 0-based! (operator[] cannot take two args)
      T operator[]( const int i ) const
      {
         return __data_[i];
      }

      // this is 0-based! (operator[] cannot take two args)
      T& restrict_ operator[]( const int i )
      {
         return __data_[i];
      }

      T operator()( const int i, const int j ) const
      {
         return __data_[(i-1)*N+(j-1)];
      }

      T& restrict_ operator()( const int i, const int j )
      {
         return __data_[(i-1)*N+(j-1)];
      }

   protected:
      T* restrict_  __data_;
};

}

#endif // __LTL_FM_ITER__
