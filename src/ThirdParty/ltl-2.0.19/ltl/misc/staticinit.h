/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: staticinit.h 491 2011-09-02 19:36:39Z drory $
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


//
// hack for being able to write MArray<float,2> A(2,2) = 1,2,3,4;
// provides overloaded oprator,() and a mechanism to
// distinguish A = something from A = 1,2,3,4;
//

#if !defined(__LTL_IN_FILE_MARRAY__) && !defined(__LTL_IN_FILE_FVECTOR__) && !defined(__LTL_IN_FILE_FMATRIX__)
#error "<ltl/misc/staticinit.h> must be included via ltl headers, never alone!"
#endif

#ifndef __LTL_STATICINIT__
#define __LTL_STATICINIT__

#include <ltl/config.h>

namespace ltl {

template<class T>
class ListInitializer
{

   public:
      ListInitializer( T* iter, int size )
            : iter_( iter ), size_(size)
      {}

      // operator,() sets the element iter points to returning a new
      // ListInitializer pointing to the next element
      // This way we work through the 'comma-delimited' list
      ListInitializer<T> operator,( T x )
      {
         LTL_ASSERT(size_>0, "List initializer past end of array");
         *iter_ = x;
         return ListInitializer<T>( iter_+1, size_-1 );
      }

   private:
      ListInitializer();

   protected:
      T*  iter_;
      int size_;
};


/*
 * Discriminate between Array=x and Array=x1,x2,x3,...
 * This class is instantiated in Array.operator=( T x ). It then decides,
 * via overloaded operator,() whether we are dealing with a list initializer
 * or simply an assignment of a scalar.
 */
template<class Array>
class ListInitializationSwitch
{

   public:
      typedef typename Array::value_type value_type;

      ListInitializationSwitch( const ListInitializationSwitch<Array>& lis )
            : array_(lis.array_), value_(lis.value_),
            fillOnDestruct(true)
      {
         lis.disable();
      }

      ListInitializationSwitch( Array& array, value_type value )
            : array_(array), value_(value), fillOnDestruct(true)
      {
#ifdef __SUNPRO_CC
         // the trick does not work with Sun's CC since it does not
         // destroy objects at the earliest possible point, rather
         // it keeps them until they go out of scope, so the destructor
         // gets calles too late.
         array_.fill(value_);
#endif
      }

      ~ListInitializationSwitch()
      {
         // here we call MArray<T,N>::fill(value). This is the case if
         // the user actually wrote A = value;
#ifndef __SUNPRO_CC
         if (fillOnDestruct)
            array_.fill(value_);
#endif
      }

      ListInitializer<value_type> operator,( value_type x )
      {
         // here we set fillOnDestruct = false since operator,() was
         // called, so we have the case A = 1,2,3,...
         fillOnDestruct = false;
         value_type* iter = array_.data();
         *iter++ = value_;  // This is the first value appearing after the operator=
         *iter++ = x;       // This is the second value after the first operator,
         // we start at the third value
         return ListInitializer<value_type>( iter, (int)array_.size()-2 );
      }

      void disable() const
      {
         fillOnDestruct = false;
      }

   private:
      ListInitializationSwitch();

   protected:
      Array&       array_;         // The container
      value_type   value_;         // The first value after operator=
      mutable bool fillOnDestruct; // Do we need to call array_.fill() because we are dealing with Array=value, not Array=x1,x2,x3,...;
};

}

#endif
