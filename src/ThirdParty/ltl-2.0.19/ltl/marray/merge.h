/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: merge.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/marray/merge.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_MERGE__
#define __LTL_MERGE__

#include <ltl/config.h>
#include <ltl/misc/type_promote.h>

namespace ltl {

/*! \defgroup merge Merge Expressions

\ingroup marray_class

For array expressions A, T, and F: \n
\code
ltl::merge( A, T, F ); // elementwise gives T if A != 0 else F.
\endcode
E.g. merge( A, 1/A, 0 ) to avoid division by 0.
*/

/// \cond DOXYGEN_IGNORE

template<class T1, class T2, class T3, int N>
class MergeExpr : public ExprBase< MergeExpr<T1,T2,T3,N>, N>, public LTLIterator
{
   public:
      typedef typename promotion_trait<typename T2::value_type,typename T3::value_type>::PType value_type;
      enum { dims=N };
      enum { numIndexIter = T1::numIndexIter + T2::numIndexIter +
                            T3::numIndexIter };
      enum { numConvolution = T1::numConvolution + T2::numConvolution +
                              T3::numConvolution };
      enum { isVectorizable = 0 };

      MergeExpr( const T1& a, const T2& e1, const T3& e2 )
            : i1_(a), i2_(e1), i3_(e2)
      { }

      MergeExpr( const MergeExpr<T1,T2,T3,N>& other )
            :  i1_(other.i1_), i2_(other.i2_), i3_(other.i3_)
      { }

      value_type operator*() const
      {
         return *i1_ ? *i2_ : *i3_;
      }

      void operator++()
      {
         ++i1_;
         ++i2_;
         ++i3_;
      }

      value_type readWithoutStride( const int i ) const
      {
         return i1_.readWithoutStride(i) ?
                i2_.readWithoutStride(i) : i3_.readWithoutStride(i);
      }

      value_type readWithStride( const int i ) const
      {
         return i1_.readWithStride(i) ?
                i2_.readWithStride(i) : i3_.readWithStride(i);
      }

      value_type readWithStride( const int i, const int dim ) const
      {
         return i1_.readWithStride(i,dim) ?
                i2_.readWithStride(i,dim) : i3_.readWithStride(i,dim);
      }

      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return i1_.readAtOffsetDim(i,dim) ? i2_.readAtOffsetDim(i,dim) : i3_.readAtOffsetDim(i,dim);
      }

      value_type readAtOffset( const int i ) const
      {
         return i1_.readAtOffset(i) ? i2_.readAtOffset(i) : i3_.readAtOffset(i);
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         return i1_.readAtOffset(i,j) ? i2_.readAtOffset(i,j) : i3_.readAtOffset(i,j);
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return i1_.readAtOffset(i,j,k) ? i2_.readAtOffset(i,j,k) : i3_.readAtOffset(i,j,k);
      }

      int boundary_l(const int dim) const
      {
         return std::min(std::min(i1_.boundary_l(dim), i2_.boundary_l(dim)), i3_.boundary_l(dim));;
      }

      int boundary_u(const int dim) const
      {
         return std::max(std::max(i1_.boundary_u(dim), i2_.boundary_u(dim)), i3_.boundary_u(dim));;
      }

      void advance()
      {
         i1_.advance();
         i2_.advance();
         i3_.advance();
      }

      void advance( const int i )
      {
         i1_.advance(i);
         i2_.advance(i);
         i3_.advance(i);
      }

      void advance( const int i, const int dim )
      {
         i1_.advance(i,dim);
         i2_.advance(i,dim);
         i3_.advance(i,dim);
      }

      void advanceWithStride1()
      {
         i1_.advanceWithStride1();
         i2_.advanceWithStride1();
         i3_.advanceWithStride1();
      }

      void advanceDim()
      {
         i1_.advanceDim();
         i2_.advanceDim();
         i3_.advanceDim();
      }

      void advanceDim(const int cutDim)
      {
         i1_.advanceDim(cutDim);
         i2_.advanceDim(cutDim);
         i3_.advanceDim(cutDim);
      }

      bool isStorageContiguous() const
      {
         return i1_.isStorageContiguous() && i2_.isStorageContiguous() &&
                i3_.isStorageContiguous();
      }

      bool isStride1() const
      {
         return i1_.isStride1() && i2_.isStride1() && i3_.isStride1();
      }

      bool isConformable( const Shape<N>& other ) const
      {
         return i1_.isConformable( other ) &&
                i2_.isConformable( other ) &&
                i3_.isConformable( other );
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
         i2_.reset();
         i3_.reset();
      }

      const Shape<N> *shape() const
      {
         return i1_.shape();
      }

   protected:
      T1 i1_;
      T2 i2_;
      T3 i3_;
};


// version for Expression, whatever, whatever
//
template<class T1, int N, class T2, class T3>
inline ExprNode< MergeExpr<typename ExprNodeType<T1>::expr_type,
                           typename ExprNodeType<T2>::expr_type,
                           typename ExprNodeType<T3>::expr_type,
                           N >,
                 N >
merge( const ExprBase<T1,N>& a, const T2& e1, const T3& e2 )
{
   typedef MergeExpr<typename ExprNodeType<T1>::expr_type,
                     typename ExprNodeType<T2>::expr_type,
                     typename ExprNodeType<T3>::expr_type,
                     N >
      ExprT;
   return ExprNode<ExprT,N>( ExprT(ExprNodeType<T1>::node(a.derived()),e1,e2) );
}

/// \endcond

}

#endif // __LTL_MERGE__
