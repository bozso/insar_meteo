/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: partial_reduc.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __LTL_PARTIAL_REDUC_H__
#define __LTL_PARTIAL_REDUC_H__

#include <ltl/config.h>

namespace ltl {

/*! \defgroup partial_reduc Partial Reductions (Contractions)

\ingroup marray_class

Contraction along a single dimension, reducing the rank by one.
*/

/// \cond DOXYGEN_IGNORE

template<typename T, typename R, int N>
class PartialReducExpr : public ExprBase<PartialReducExpr<T,R,N>, N >, public LTLIterator
{
   public:
      typedef typename R::value_type value_type;
      enum { dims=N };
      enum { numIndexIter = 1 };   // force evaluation without collapsing loops.
      enum { numConvolution = 0 };
      enum { isVectorizable = 0 };

      PartialReducExpr( const T& a, int dim )
            : i1_(a), dim_(dim-1), len_(a.shape()->length(dim)), shape_(a.shape()->getShapeForContraction(dim))
      { }

      PartialReducExpr( const PartialReducExpr<T,R,N>& other )
            :  i1_(other.i1_), dim_(other.dim_), len_(other.len_), shape_(other.shape_)
      { }

      void operator++()
      {
         ++i1_;
      }

      value_type operator*() const
      {
         return eval();
      }

      value_type readWithoutStride( const int i ) const
      {
         return i1_.readWithoutStride(i);
      }

      value_type readWithStride( const int i ) const
      {
         return i1_.readWithStride(i);
      }

      value_type readWithStride( const int i, const int dim ) const
      {
         return i1_.readWithStride(i, dim);
      }

      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return i1_.readAtOffsetDim(i,dim);
      }

      value_type readAtOffset( const int i ) const
      {
         return i1_.readAtOffset(i);
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         return i1_.readAtOffset(i,j);
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return i1_.readAtOffset(i,j,k);
      }

      int boundary_l(const int dim) const
      {
         return i1_.boundary_l(dim);
      }

      int boundary_u(const int dim) const
      {
         return i1_.boundary_u(dim);
      }

      void advance()
      {
         advance( 1, (0==dim_ ? 1 : 0) );
      }

      void advance( const int i )
      {
         advance( i, (0==dim_ ? 1 : 0) );
      }

      void advanceWithStride1()
      {
         advance( 1, (0==dim_ ? 1 : 0) );
      }

      void advance( const int i, const int dim )
      {
         i1_.advance(i, dim==dim_ ? dim+1 : dim);
      }

      void advanceDim()
      {
         i1_.advanceDim(dim_);
      }

      bool isStorageContiguous() const
      {
         return i1_.isStorageContiguous();
      }

      bool isStride1() const
      {
         return i1_.isStride1();
      }

      bool isConformable( const Shape<N>& other ) const
      {
         // our shape (after contraction) has to be conformable with whatever expression we are
         // part of; and the expression we are reducing has to have conformable operands itself.
         return shape_.isConformable( other ) && i1_.isConformable( *i1_.shape() );
      }

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline T_VEC_TYPE(value_type) readVec( const int i ) const
      {
         return (T_VEC_TYPE(value_type))(0);
      }

      inline int getAlignment() const
      {
         return 0;
      }

      inline void alignData( const int align )
      {
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return false;
      }
#endif

     void reset()
      {
         i1_.reset();
      }

      const Shape<N> *shape() const
      {
         return &shape_;
      }

   protected:

      value_type eval() const
      {
         R R_;
         int j = 0, innerLoopNum = len_;

#ifdef LTL_UNROLL_EXPRESSIONS_1
         for(; j<innerLoopNum-3; j+=4 )
         {
            if( !R_.evaluate( i1_.readWithStride(j, dim_) ) )
               goto loop_end;
            if( !R_.evaluate( i1_.readWithStride(j+1, dim_) ) )
               goto loop_end;
            if( !R_.evaluate( i1_.readWithStride(j+2, dim_) ) )
               goto loop_end;
            if( !R_.evaluate( i1_.readWithStride(j+3, dim_) ) )
               goto loop_end;
         }
#endif
         for(; j<innerLoopNum; ++j )
         {
            //cout << j << " " << i1_.readWithStride(j, dim_) << endl;
            if( !R_.evaluate( i1_.readWithStride(j, dim_) ) )
               goto loop_end;
         }

         loop_end:
            return R_.result();
      }

      T i1_;
      const int dim_;
      const int len_;
      const Shape<N> shape_;
};

/// \endcond


#define DECLARE_PARTIAL_REDUCTION(func,reduction)                                            \
template<class T, int N>                                                                     \
inline ExprNode<PartialReducExpr<typename ExprNodeType<T>::expr_type,                        \
                                 reduction<typename T::value_type>,                          \
                                 N-1>,                                                       \
                N-1>                                                                         \
partial_##func ( const ExprBase<T,N>& A, const int M )                                       \
{                                                                                            \
   LTL_ASSERT( N>1, "Can't partially reduce 1-dimensional MArray" );                         \
   LTL_ASSERT( M>0 && M<=N, "Bad dimension "<<M<<" in partial reduction of dimension "<<N ); \
   typedef PartialReducExpr<typename ExprNodeType<T>::expr_type,                             \
                            reduction<typename T::value_type>,                               \
                            N-1>                                                             \
      ExprT;                                                                                 \
   return ExprNode<ExprT,N-1>( ExprT(ExprNodeType<T>::node(A.derived()),M) );                \
}                                                                                            \

DECLARE_PARTIAL_REDUCTION( allof, allof_reduction )
DECLARE_PARTIAL_REDUCTION( noneof, noneof_reduction )
DECLARE_PARTIAL_REDUCTION( anyof, anyof_reduction )
DECLARE_PARTIAL_REDUCTION( count, count_reduction )

DECLARE_PARTIAL_REDUCTION( min, min_reduction )
DECLARE_PARTIAL_REDUCTION( max, max_reduction )

DECLARE_PARTIAL_REDUCTION( sum, sum_reduction )
DECLARE_PARTIAL_REDUCTION( product, prod_reduction )

DECLARE_PARTIAL_REDUCTION( average, avg_reduction )
DECLARE_PARTIAL_REDUCTION( variance, variance_reduction )

}

#endif // __LTL_PARTIAL_REDUC_H__
