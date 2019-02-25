/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: apply.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C) Niv Drory <drory@mpe.mpg.de.de>
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
#error "<ltl/marray/apply.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_APPLY__
#define __LTL_APPLY__

namespace ltl {

/*! \defgroup apply Apply Expressions

\ingroup marray_class

Apply a function object (any object that defines operator()) to
all elements of an MArray or Expression.

For array expression E and function object F \n
\code
A = ltl::apply( F, E ); // elementwise apply F.operator() to E.
A = ltl::apply( G, A, B ); // same, with a binary functor: apply G.operator() to A,B
\endcode

The functor F has to define \code typedef typename F::value_type\endcode
and if it is to be used in \c LTL_USE_SIMD mode, it has to provide
\code enum { isVectorizable = 0/1 }\endcode to indicate if it is
vectorizable or not. If it is vectorizable, it has to provide
\code operator()( VEC_TYPE(parameter_type) )\endcode which will be used
during evaluation.

Here's an example:

\code
struct functor
{
   typedef float value_type;
   enum { isVectorizable = 0 };

   value_type operator()( float a )
   {
      return a*a;
   }
};

struct binary_functor
{
   typedef float value_type;
   enum { isVectorizable = 0 };

   value_type operator()( float a, float b )
   {
      return a+b;
   }
};
\endcode

Here's the implementation of the above unary functor in a vectorizable version:

\code
struct functor
{
   typedef float value_type;
   enum { isVectorizable = 1 };

   float operator()( float a )
   {
      return a*a;
   }

   // SSE implementation of float multiply:
   VEC_TYPE(float) operator()( VEC_TYPE(float) a )
   {
      return _mm_mul_ps(a,a);
   }
};

\endcode
*/

/// \cond DOXYGEN_IGNORE
//
// unary functors
//

template<typename F, typename E, int N>
class ApplyExpr : public ExprBase< ApplyExpr<F,E,N>, N >, public LTLIterator
{
   public:
      typedef typename F::value_type value_type;
      enum { dims=N };
      enum { numIndexIter = E::numIndexIter };
      enum { numConvolution = E::numConvolution };
      enum { isVectorizable = E::isVectorizable && F::isVectorizable };

      ApplyExpr( const F& f, const E& e1 )
            : F_(f), E_(e1)
      { }

      ApplyExpr( const ApplyExpr<F,E,N>& other )
            :  F_(other.F_), E_(other.E_)
      { }

      value_type operator*() const
      {
         return const_cast<F&>(F_)(*E_);
      }

      void operator++()
      {
         ++E_;
      }

      value_type readWithoutStride( const int i ) const
      {
         return const_cast<F&>(F_)( E_.readWithoutStride(i) );
      }

      value_type readWithStride( const int i ) const
      {
         return const_cast<F&>(F_)( E_.readWithStride(i) );
      }

      value_type readWithStride( const int i, const int dim ) const
      {
         return const_cast<F&>(F_)( E_.readWithStride(i,dim) );
      }

      value_type readAtOffsetDim( const int i, const int dim )
      {
         return const_cast<F&>(F_)( E_.readAtOffsetDim(i,dim) );
      }

      value_type readAtOffset( const int i ) const
      {
         return const_cast<F&>(F_)( E_.readAtOffset(i) );
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         return const_cast<F&>(F_)( E_.readAtOffset(i,j) );
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return const_cast<F&>(F_)( E_.readAtOffset(i,j,k) );
      }

      int boundary_l(const int dim) const
      {
         return E_.boundary_l(dim);
      }

      int boundary_u(const int dim) const
      {
         return E_.boundary_u(dim);
      }

      void advance()
      {
         E_.advance();
      }

      void advance( const int i )
      {
         E_.advance(i);
      }

      void advance( const int i, const int dim )
      {
         E_.advance(i,dim);
      }

      void advanceWithStride1()
      {
         E_.advanceWithStride1();
      }

      void advanceDim()
      {
         E_.advanceDim();
      }

      void advanceDim(const int cutDim)
      {
         E_.advanceDim(cutDim);
      }

      bool isStorageContiguous() const
      {
         return E_.isStorageContiguous();
      }

      bool isStride1() const
      {
         return E_.isStride1();
      }

      bool isConformable( const Shape<N>& other ) const
      {
         return E_.isConformable( other );
      }

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      {
         return const_cast<F&>(F_)( E_.readVec(i) );
      }

      inline void alignData( const int align )
      {
         E_.alignData( align );
      }

      inline int getAlignment() const
      {
         return E_.getAlignment();
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return E_.sameAlignmentAs(p);
      }
#endif

     void reset()
      {
         E_.reset();
      }

      const Shape<N> *shape() const
      {
         return E_.shape();
      }

   protected:
      F F_;
      E E_;
};


// global apply() function template for unary functors.
// this template takes \c ExprBase objects and
// returns an \c ExprNode
//
template<typename F, typename T1, int N>
inline ExprNode< ApplyExpr<F, typename ExprNodeType<T1>::expr_type, N>, N >
apply( F& f, const ExprBase<T1,N>& A )
{
   typedef ApplyExpr<F, typename ExprNodeType<T1>::expr_type, N> ExprT;
   return ExprNode<ExprT,N>( ExprT( f, ExprNodeType<T1>::node(A.derived()) ) );
}



//
// now for binary functors
//

template<typename F, typename E1, typename E2, int N>
class ApplyExprBin : public ExprBase<ApplyExprBin<F,E1,E2,N>, N >
{
   public:
      typedef typename F::value_type value_type;
      enum { dims=N };
      enum { numIndexIter = E1::numIndexIter + E2::numIndexIter };
      enum { numConvolution = E1::numConvolution + E2::numConvolution };
      enum { isVectorizable = E1::isVectorizable && E2::isVectorizable && F::isVectorizable };

      ApplyExprBin( const F& f, const E1& e1, const E2& e2 )
            : F_(f), E1_(e1), E2_(e2)
      { }

      ApplyExprBin( const ApplyExprBin<F,E1,E2,N>& other )
            :  F_(other.F_), E1_(other.E1_), E2_(other.E2_)
      { }

      value_type operator*() const
      {
         return const_cast<F&>(F_)(*E1_,*E2_);
      }

      void operator++()
      {
         ++E1_;
         ++E2_;
      }

      value_type readWithoutStride( const int i ) const
      {
         return const_cast<F&>(F_)( E1_.readWithoutStride(i),E2_.readWithoutStride(i) );
      }

      value_type readWithStride( const int i ) const
      {
         return const_cast<F&>(F_)( E1_.readWithStride(i),E2_.readWithStride(i) );
      }

      value_type readWithStride( const int i, const int dim ) const
      {
         return const_cast<F&>(F_)( E1_.readWithStride(i,dim),E2_.readWithStride(i,dim) );
      }

      value_type readAtOffset( const int i ) const
      {
         return const_cast<F&>(F_)( E1_.readAtOffset(i), E2_.readAtOffset(i) );
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         return const_cast<F&>(F_)( E1_.readAtOffset(i,j), E2_.readAtOffset(i,j) );
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return const_cast<F&>(F_)( E1_.readAtOffset(i,j,k), E2_.readAtOffset(i,j,k) );
      }

      int boundary_l(const int dim) const
      {
         return std::min(E1_.boundary_l(dim), E2_.boundary_l(dim));
      }

      int boundary_u(const int dim) const
      {
         return std::max(E1_.boundary_u(dim), E2_.boundary_u(dim));
      }

      void advance()
      {
         E1_.advance();
         E2_.advance();
      }

      void advance( const int i )
      {
         E2_.advance(i);
         E2_.advance(i);
      }

      void advance( const int i, const int dim )
      {
         E1_.advance(i,dim);
         E2_.advance(i,dim);
      }

      void advanceWithStride1()
      {
         E1_.advanceWithStride1();
         E2_.advanceWithStride1();
      }

      void advanceDim()
      {
         E1_.advanceDim();
         E2_.advanceDim();
      }

      void advanceDim(const int cutDim)
      {
         E1_.advanceDim(cutDim);
         E2_.advanceDim(cutDim);
      }

      bool isStorageContiguous() const
      {
         return E1_.isStorageContiguous() && E2_.isStorageContiguous();
      }

      bool isStride1() const
      {
         return E1_.isStride1() && E2_.isStride1();
      }

      bool isConformable( const Shape<N>& other ) const
      {
         return E1_.isConformable( other ) && E2_.isConformable( other );
      }

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      {
         return const_cast<F&>(F_)( E1_.readVec(i),E2_.readVec(i) );
      }

      inline void alignData( const int align )
      {
         E1_.alignData( align );
         E2_.alignData( align );
      }

      inline int getAlignment() const
      {
         return _expr_getalign( E1_, E2_ );
      }

      inline bool sameAlignmentAs( void* p ) const
      {
         return E1_.sameAlignmentAs(p) && E2_.sameAlignmentAs(p);
      }
#endif

     void reset()
      {
        E1_.reset();
        E2_.reset();
      }

      const Shape<N> *shape() const
      {
         return E1_.shape();
      }

   protected:
      F  F_;
      E1 E1_;
      E2 E2_;
};


// global apply() function template for binary functors.
// this template takes \c ExprBase objects and
// returns an \c ExprNode
//
template<typename F, typename T1, typename T2, int N>
inline ExprNode< ApplyExprBin<F, typename ExprNodeType<T1>::expr_type, typename ExprNodeType<T2>::expr_type, N>, N >
apply( F& f, const ExprBase<T1,N>& A, const ExprBase<T2,N>& B )
{
   typedef ApplyExprBin<F, typename ExprNodeType<T1>::expr_type, typename ExprNodeType<T2>::expr_type, N> ExprT;
   return ExprNode<ExprT,N>( ExprT( f, ExprNodeType<T1>::node(A.derived()), ExprNodeType<T2>::node(B.derived()) ) );
}

/// \endcond
}

#endif // __LTL_APPLY__
