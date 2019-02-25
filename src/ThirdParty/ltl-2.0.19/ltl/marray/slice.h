/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: slice.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/marray/slice.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_SLICE__
#define __LTL_SLICE__

/*! \file slice.h
  Very weird, but working solution for the problem of having MArrays being
  constructed by providing a mixture of Range and int (Slice) arguments.
  Somehow, counting the number of int arguments has to be done at compile-
  time, to get the right template<Type,Dim> dimension argument for the
  newly constructed MArray object.

  The idea is based on enum computations: the class SliceCounter builds
  up an enum type, adding 1 for every Range argument. Then a typedef
  is made to MAarray<>.
*/

#include <ltl/config.h>

namespace ltl {

template<class T, int D>
class MArray;


class NoArgument
{ };  // dummy for default arguments

template<class T>
class RangeSliceArgument
{
   public:
      enum { isValid=0, dim=0};
};

template<>
class RangeSliceArgument<Range>
{
   public:
      enum { isValid=1, dim=1};  // Range() argument keeps dimension
};

template<>
class RangeSliceArgument<int>
{
   public:
      enum { isValid=1, dim=0};  // int argument makes a slice => dimension is
      // removed
};

template<>
class RangeSliceArgument<NoArgument>
{
   public:
      enum { isValid=1, dim=0};  // placeholder for missing arguments
};


template<class Type, class T1, class T2=NoArgument, class T3=NoArgument,
         class T4=NoArgument, class T5=NoArgument, class T6=NoArgument, 
         class T7=NoArgument >
class SliceCounter
{
   public:
      enum {
         dim      = RangeSliceArgument<T1>::dim
                    + RangeSliceArgument<T2>::dim
                    + RangeSliceArgument<T3>::dim
                    + RangeSliceArgument<T4>::dim
                    + RangeSliceArgument<T5>::dim
                    + RangeSliceArgument<T6>::dim
                    + RangeSliceArgument<T7>::dim
   };

      typedef MArray<Type,dim>      MArraySlice;   // will become lhs

};

}

#endif
