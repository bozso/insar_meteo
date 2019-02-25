/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: expr_iter.h 541 2014-07-09 17:01:12Z drory $
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

/*! \class ltl::ExprIter
  This is a special version of an iterator useful for iterating over
  a ltl::ExprNode<A,N> object, which is an iterator by itself, but which would
  be very inefficient to use directly, since it delegates ALL calls to
  ALL subnodes in an expression parse-tree. Here we define an ExprIter
  object which uses the 'remote-control' mechanism of the expr to iterate
  efficiently. See also comments in marray_iter.h and eval.h.
*/

#ifndef __LTL_MARRAY__
#error "<ltl/marray/expr_iter.h> must be included via <ltl/marray.h>, never alone!"
#endif



#ifndef __LTL_EXPR_ITER__
#define __LTL_EXPR_ITER__

#include <ltl/config.h>
#include <ltl/marray/shape_iter.h>

namespace ltl {

template<typename A, int N>
class ExprIter
{
   public:
      typedef std::forward_iterator_tag       iterator_category;
      typedef typename ExprNode<A,N>::value_type value_type;
      typedef          int                    difference_type;
      typedef const value_type&               const_reference;
      typedef const value_type*               const_pointer;
      typedef       value_type&               reference;
      typedef       value_type*               pointer;

      enum { isVectorizable = ExprNode<A,N>::isVectorizable };


      ExprIter( ExprNode<A,N>& e )
            : e_(e), i_(*e.shape())
      { }

      ExprIter( ExprNode<A,N>& e, bool )
            : e_(e), i_()
      { }

      ExprIter( const ExprIter<A,N>& other )
            : e_(other.e_), i_(other.i_)
      { }

      inline ExprIter<A,N>& operator++()
      {
         i_.advance();
         e_.advance();
         if( i_.needAdvanceDim() )
         {
            i_.advanceDim();
            e_.advanceDim();
         }
         return *this;
      }

      ExprIter<A,N> operator++( int )
      {
         ExprIter<A,N> tmp( *this );
         ++(*this);
         return tmp;
      }

      inline bool done() const
      {
         return i_.done();
      }

      inline value_type operator*() const
      {
         return *e_;
      }

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      {
         return e_.readVec(i);
      }

      inline void alignData( const int align )
      {
         e_.alignData( align );
      }

      inline int getAlignment() const
      {
         return e_.getAlignment();
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return e_.sameAlignmentAs(p);
      }
#endif

      //! FUDGE! but works for comparison with end iterator in STL algorithms
      inline bool operator==( const ExprIter<A,N>& ) const
      {
         return done();
      }

      //! FUDGE! but works for comparison with end iterator in STL algorithms
      inline bool operator!=( const ExprIter<A,N>& ) const
      {
         return !done();
      }

      //! inquire about the length of the loop over each dimension
      int length( const int i ) const
      {
         return i_.length(i);
      }

      const Shape<N>* shape() const
      {
         return i_.shape();
      }

   protected:
      ExprNode<A,N>& e_;
      ShapeIter<N> i_;
};

}

#endif

