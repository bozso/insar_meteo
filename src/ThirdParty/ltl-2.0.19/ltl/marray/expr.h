/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: expr.h 541 2014-07-09 17:01:12Z drory $
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
#error "<ltl/marray/expr.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_EXPR__
#define __LTL_EXPR__


namespace ltl {

/// \cond DOXYGEN_IGNORE
template<class T, int D>
class MArray;

template<class T, int N>
class IndexIterator;

template<int N>
class Shape;

template<class A, class B, class Op, int N>
class ExprBinopNode;

template<class A, class Op, int N>
class ExprUnopNode;

template<class T>
class ExprLiteralNode;

template<class A, int N>
class ExprIter;
/// \endcond

/*! \file expr.h
 *  \brief This file defines expression template parse tree nodes and macors for creating global unary and binary operators/functions.
 *
 *  \defgroup marray_expr MArray Expression Template Internals
 *
 *  \ingroup marray_class
 *
 *  This documentation explains the internal expression template mechanism. It should not be necessary to understand all this
 *  to use the LTL.
 *
 *  The anatomy of expression templates:
 *
 *  Expressions like the right-hand side of
 *  \code
 *     A = A + B*C;
 *  \endcode
 *  (with \c A, \c B, and \c C being of type \c ltl::MArray<float,1>)
 *  are represented by nested templated data types that capture the parse tree of the expression.
 *
 *  In this example, the expression A + B*C is represented by the type (simplified for readability)
 *  \code
 *     ltl::ExprBinopNode<ltl::MArrayIterConst<float, 1>,
 *                        ltl::ExprNode<ltl::ExprBinopNode<ltl::MArrayIterConst<float, 1>,
 *                                                         ltl::MArrayIterConst<float, 1>,
 *                                                         ltl::__ltl_TMul<float, float>,
 *                                                         1>,
 *                                      1>,
 *                        ltl::__ltl_TAdd<float, float>,
 *                        1>
 *  \endcode
 *  In real code, we'd want all possible parse tree nodes to be of one single type so that any expression
 *  argument can be caught by a single templated type. Therefore, each of the parse tree nodes is wrapped
 *  in a template class \c ExprNode<>, which forwards all calls to the parse tree node it wraps.
 *
 *  The class \c ExprNode<> defines an iterator-like interface that all parse tree nodes (and ultimately
 *  the iterators that parse tree nodes hold) implement. This way, a whole expression, and every sub-expression
 *  presents an iterator interface to the outside world.
 *
 *  For this to function we define
 *    \li A global-scope overloaded function or operator for each function or operator we wish to use in expression templates
 *    e.g., \c operator+ and \c sin(). This function/operator takes types representing nodes in the parse tree as arguments
 *    and return a type representing their own node in the parse tree.
 *    For example \c opertator+ for two expressions:
 *    \code
 *      template<typename A, typename B, int N>
 *      inline ExprNode<ExprBinopNode<ExprNode<A,N>,
 *                                    ExprNode<B,N>,
 *                                    __ltl_TAdd<typename A::value_type, typename B::value_type>,
 *                                    N >, N >
 *      operator+( const ExprNode<A,N>& a, const ExprNode<B,N>& b )
 *      {
 *         typedef ExprBinopNode<ExprNode<A,N>,
 *                               ExprNode<B,N>,
 *                               __ltl_TAdd<typename A::value_type, typename B::value_type>, N >
 *         ExprT;
 *         return ExprNode<ExprT,N>( ExprT(a,b) );
 *      }
 *    \endcode
 *    Since we need overloaded versions for all operations, we will define macros to help set
 *    these up.
 *
 *    \li Each of these global functions encodes the operation it represents by a further template
 *    parameter of the parse tree node. This is a struct (called an applicative template) that
 *    implements the operation on scalar types via a method called \c eval(). In the above example these
 *    are \c ltl::__ltl_TMul and \c ltl::__ltl_TAdd.
 *    \c ltl::__ltl_TAdd looks like this:
 *    \code
 *      template<typename T1, typename T2>
 *      struct __ltl_TAdd : public _et_applic_base
 *      {
 *         typedef typename promotion_trait<T1,T2>::PType value_type;
 *         static inline value_type eval( const T1& a, const T2& b )
 *         { return a + b; }
 *      }
 *    \endcode
 *
 *    \li Nodes in the parse tree implement the same interface as ltl::MArrayIter iterators.
 *
 *  Ultimately, the expression is evaluated by
 *  \code
 *    template<typename E>
 *    ltl::MArray<float,1>::operator=( ExprNode<E,1> ).
 *  \endcode
 *  which uses the iterator-like interface of the expression to loop over each element of the underlying
 *  (multidimensional) array structure.
 *
 * TODO: properly document the new ET mechanism, and also all types derived from ExprBase,
 * e.g. MergeExpr, ... and how to add user-defined expressions.
 */

/*! Determine the \c shape of an expression
 *  \ingroup marray_expr
 *
 *  When determining the \c shape of an expression it is sufficient to
 *  return any one \c shape object from any one of the MArrays in the expression
 *  since we know they will all be the conformable. However, we must make sure
 *  that we do not ask e.g. a literal constant for a shape ...
 *
 *  In the general case, just use the LHS Shape ... We will provide partial
 *  specializations for the cases where one of the operands does not have a
 *  shape and return the other instead.
 */
//! Determine the \c shape of an expression by returning the ltl::Shape objects on one of the ltl::MArray operatnds.
template<typename A, typename B>
inline const Shape<A::dims>*
_expr_getshape( const A& a, const B& )
{
   return a.shape();
}
// Specialization for literals: The LHS is an expression and the RHS is a literal
template<typename A, typename T>
inline const Shape<A::dims>*
_expr_getshape( const A& a, const ExprLiteralNode<T>& )
{
   return a.shape();
}
// Specialization for literals: The RHS is an expression and the LHS is a literal
template<typename A, typename T>
inline const Shape<A::dims>*
_expr_getshape( const ExprLiteralNode<T>&, const A& a )
{
   return a.shape();
}


/*! \ingroup marray_expr
 *
 *  When determining the alignment of an expression it is sufficient to
 *  return any one of the alignments from any one of the MArrays in the expression
 *  since we only vectorize if we know the alignments are the same. However, we must make sure
 *  that we do not ask e.g. a literal constant for a it's alignment ...
 *
 *  In the general case, just use the LHS's alignment ... We will provide partial
 *  specializations for the cases where one of the operands does not have an alignment
 *  and return the other's instead.
 */
//! Determine the alignment (w.r.t. natural vector boundaries) of the operands in an expression
template<typename A, typename B>
inline int
_expr_getalign( const A& a, const B& )
{
   return a.getAlignment();
}

// The LHS is an expression and the RHS is a literal
template<typename A, typename T>
inline int
_expr_getalign( const A& a, const ExprLiteralNode<T>& )
{
   return a.getAlignment();
}
// The RHS is an expression and the LHS is a literal
template<typename T, typename A>
inline int
_expr_getalign( const ExprLiteralNode<T>&, const A& a )
{
   return a.getAlignment();
}


//! Node in the expression parse tree. Every expression in ultimately represented by this class.
/*!
 *  \ingroup marray_expr
 *  This class represents a node in the parse tree of an expression. Any operation or
 *  operand is captured as a type \c ExprNode<Op,N>, where \c Op represents the operation
 *  or operand and N the number of dimensions. This way, there is a single
 *  data type \c ExprNode<> associated with any parse tree element.
 *
 *  The operation or operand \c Op are of the types
 *    \c ExprLiteralNode
 *    \c ExprBinopNode
 *    \c ExprUnopNode
 *    \c MArrayIter
 *    \c MergeExpr
 *    \c ApplyExpr
 *    \c ApplyExprBin
 *    \c ConvolveExpr
 *
 *  Like all parse tree elements, this class implements the iterator interface. It forwards
 *  all iterator calls to the operation or operand it wraps.
 */
template<typename A, int N>
class ExprNode : public ExprBase<ExprNode<A,N>, N>, public LTLIterator
{
   private:
      A iter_;  //<! The operation or operand, itself implementing the iterator interface. All operations will be forwarded to this object.

   public:

      //! the result data type of the parse tree node
      typedef typename A::value_type value_type;

      //@{
      //! The number of dimensions
      enum { dims=N };
      //! The number of ltl::IndexIter index iterators in all the parse tree below this node
      enum { numIndexIter = A::numIndexIter };
      //! The number of convolution operations in the parse tree below this node
      enum { numConvolution = A::numConvolution };
      //! Is the whole parse tree below this node vectorizable
      enum { isVectorizable = A::isVectorizable };
      //@}

      //! Constructor
      inline ExprNode( const A& a )
            : iter_(a)
      { }

      //! Move all iterators in the parse tree below us forward. Slowest, but works for all expressions.
      inline void operator++()
      {
         ++iter_;
      }

      //! compute and return the value of the subexpression in the parse tree below us.
      inline value_type operator*() const
      {
         return *iter_;
      }

      //@{
      /*!
       *  Move all iterators in the parse tree below us forward.
       *  Used in unrolling/combining the expression evaluation loops when possible.
       */
      inline void advance()
      {
         iter_.advance();
      }
      inline void advance( const int i )
      {
         iter_.advance(i);
      }
      inline void advance( const int i, const int dim )
      {
         iter_.advance(i,dim);
      }
      inline void advanceWithStride1()
      {
         iter_.advanceWithStride1();
      }
      inline void advanceDim()
      {
         iter_.advanceDim();
      }
      inline void advanceDim(const int cutDim)
      {
         iter_.advanceDim(cutDim);
      }
      //@}

      //@{
      /*!
       *  Compute and return the value of the subexpression in the parse tree below us.
       *  Used in unrolling/combining the expression evaluation loops when possible.
       */
      inline value_type readWithoutStride( const int i ) const
      {
         return iter_.readWithoutStride(i);
      }
      inline value_type readWithStride( const int i ) const
      {
         return iter_.readWithStride(i);
      }
      inline value_type readWithStride( const int i, const int dim ) const
      {
         return iter_.readWithStride(i, dim);
      }
      //@}

      //@{
      /*!
       *  Compute and return the value of the subexpression in the parse tree below us
       *  at an offset in dimension 1, 2, and/or 3. Used for evaluating convolutions
       */
      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return iter_.readAtOffsetDim(i,dim);
      }
      value_type readAtOffset( const int i ) const
      {
         return iter_.readAtOffset(i);
      }
      value_type readAtOffset( const int i, const int j ) const
      {
         return iter_.readAtOffset(i,j);
      }
      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return iter_.readAtOffset(i,j,k);
      }
      //@}

      //@{
      /*!
       *  Return the width of the lower/upper boundary in dimension \c dim to be left unevaluated.
       *  Zero in most cases except for convolutions, which do require padding of the half-width
       *  of the kernel.
       */
      int boundary_l(const int dim) const
      {
         return iter_.boundary_l(dim);
      }
      int boundary_u(const int dim) const
      {
         return iter_.boundary_u(dim);
      }
      //@}

#ifdef LTL_USE_SIMD
      //@{
      //! Vectorization interface
      typedef typename A::vec_value_type vec_value_type;
      inline typename VEC_TYPE(vec_value_type) readVec( const int i ) const
      {
         return iter_.readVec(i);
      }

      inline void alignData( const int align )
      {
         iter_.alignData( align );
      }

      inline int getAlignment() const
      {
         return iter_.getAlignment();
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return iter_.sameAlignmentAs(p);
      }
      //@}
#endif

      //@{
      //! Storage geometry
      inline bool isStorageContiguous() const
      {
         return iter_.isStorageContiguous();
      }
      //! true if the innermost stride is unity
      inline bool isStride1() const
      {
         return iter_.isStride1();
      }
      //! true if we are conformable with another shape
      inline bool isConformable( const Shape<N>& other ) const
      {
         return iter_.isConformable( other );
      }
      //! Reset the iterators in the parse tree below us
      inline void reset()
      {
         iter_.reset();
      }
      //! Return one of the \c shape objects from the parse tree
      const Shape<N> *shape() const
      {
         return iter_.shape();
      }
      //@}

      //@{
      //! Return an \c ExprIter
      inline ExprIter<A,N> begin()
      {
         return ExprIter<A,N>( *this );
      }
      //! Return an end \c ExprIter
      inline ExprIter<A,N> end()
      {
         return ExprIter<A,N>( *this, true );
      }
      //@}
};


//! Binary operation node in the expression parse tree
/*!
 *  \ingroup marray_expr
 *  This class represents a binary operation in the parse tree of an expression.
 *  It captures the LHS and the RHS of the operation (both of type \c ExprNode<>)
 *  and the operation itself, which which is a functor that encapsulates the operation for the
 *  element type of the arrays/expressions involved.
 *
 *  Like all parse tree elements, this class implements the iterator interface. It forwards
 *  all iterator movements to the LHS and the RHS, and when dereferenced, evaluates the operation
 *  passing the LHS and RHS as parameters.
 *
 *  See the documentation of \ref ExprNode for an anatomy of the iterator interface.
 */
template<typename A, typename B, typename Op, int N>
class ExprBinopNode: public LTLIterator
{
   private:
      A iter1_;  //!< The LHS of the binary operation
      B iter2_;  //!< The RHS of the binary operation

   public:
      //! the result data type is the \c value_type of the operation
      typedef typename Op::value_type value_type;

      //@{
      //! The number of dimensions
      enum { dims=N };
      //! The number of ltl::IndexIter index iterators in the LHS and the RHS
      enum { numIndexIter = A::numIndexIter + B::numIndexIter };
      //! The number of convolution operations
      enum { numConvolution = A::numConvolution + B::numConvolution };
      //! Vectorizable if the LHS and RHS are vectorizable and if the the operation is vectorizable
      enum { isVectorizable = A::isVectorizable * B::isVectorizable * Op::isVectorizable };
      //@}

      //! Constructor
      inline ExprBinopNode( const A& a, const B& b )
            : iter1_(a), iter2_(b)
      { }

      //@{
      //! Implement the iterator interface forwarding all operations to both operands
      inline void operator++()
      {
         ++iter1_;
         ++iter2_;
      }

      inline void advance()
      {
         iter1_.advance();
         iter2_.advance();
      }

      inline void advance( const int i )
      {
         iter1_.advance(i);
         iter2_.advance(i);
      }

      inline void advance( const int i, const int dim )
      {
         iter1_.advance(i,dim);
         iter2_.advance(i,dim);
      }

      inline void advanceWithStride1()
      {
         iter1_.advanceWithStride1();
         iter2_.advanceWithStride1();
      }

      inline void advanceDim()
      {
         iter1_.advanceDim();
         iter2_.advanceDim();
      }

      inline void advanceDim(const int cutDim)
      {
         iter1_.advanceDim(cutDim);
         iter2_.advanceDim(cutDim);
      }
      //@{
      //! Evaluate by passing the values of the LHS and RHS to the operation
      inline value_type operator*() const
      {
         return Op::eval( *iter1_, *iter2_ );
      }

      inline value_type readWithoutStride( const int i ) const
      {
         return Op::eval( iter1_.readWithoutStride(i),
                          iter2_.readWithoutStride(i) );
      }

      inline value_type readWithStride( const int i ) const
      {
         return Op::eval( iter1_.readWithStride(i),
                          iter2_.readWithStride(i) );
      }

      inline value_type readWithStride( const int i, const int dim ) const
      {
         return Op::eval( iter1_.readWithStride(i, dim),
                          iter2_.readWithStride(i, dim) );
      }
      //@}

      //@{
      /*! compute and return the value of the subexpression in the parse tree below us
       *  at an offset in dimension 1, 2, and/or 3. Used for evaluating convolutions
       */
      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return Op::eval(iter1_.readAtOffsetDim(i,dim), iter2_.readAtOffsetDim(i,dim));
      }
      value_type readAtOffset( const int i ) const
      {
         return Op::eval(iter1_.readAtOffset(i), iter2_.readAtOffset(i));
      }
      value_type readAtOffset( const int i, const int j ) const
      {
         return Op::eval(iter1_.readAtOffset(i,j), iter2_.readAtOffset(i,j));
      }
      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return Op::eval(iter1_.readAtOffset(i,j,k), iter2_.readAtOffset(i,j,k));
      }
      //@}

      //@{
      /*!
       *  Return the width of the lower/upper boundary in dimension \c dim to be left unevaluated.
       *  Zero in most cases except for convolutions, which do require padding of the half-width
       *  of the kernel.
       */
      int boundary_l(const int dim) const
      {
         return std::min(iter1_.boundary_l(dim), iter2_.boundary_l(dim));
      }
      int boundary_u(const int dim) const
      {
         return std::max(iter1_.boundary_u(dim), iter2_.boundary_u(dim));
      }
      //@}

#ifdef LTL_USE_SIMD
      typedef typename Op::vec_value_type vec_value_type;
      inline typename VEC_TYPE(vec_value_type) readVec( const int i ) const
      {
         return Op::eval_vec( iter1_.readVec(i), iter2_.readVec(i) );
      }

      inline void alignData( const int align )
      {
         iter1_.alignData( align );
         iter2_.alignData( align );
      }

      inline int getAlignment() const
      {
         return _expr_getalign( iter1_, iter2_ );
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return iter1_.sameAlignmentAs(p) && iter2_.sameAlignmentAs(p);
      }
#endif

      inline bool isStorageContiguous() const
      {
         return iter1_.isStorageContiguous() && iter2_.isStorageContiguous();
      }

      inline bool isStride1() const
      {
         return iter1_.isStride1() && iter2_.isStride1();
      }

      inline bool isConformable( const Shape<N>& other ) const
      {
         return iter1_.isConformable( other ) &&
                iter2_.isConformable( other );
      }

      inline void reset()
      {
         iter1_.reset();
         iter2_.reset();
      }

      //! Return a \c shape object from the parse tree
      inline const Shape<N> *shape() const
      {
         return _expr_getshape( iter1_, iter2_ );
      }
      //@}
};


//! Unary operation node in the expression parse tree
/*!
 *  \ingroup marray_expr
 *  This class represents a unary operation in the parse tree of an expression.
 *  It captures the operrand of the operation (of type \c ExprNode<>)
 *  and the operation itself, which is a functor that encapsulates the operation for the
 *  element type of the arrays/expressions involved.
 *
 *  Like all parse tree elements, this class implements the iterator interface. It forwards
 *  all iterator movements to the operand, and when dereferenced, evaluates the operation
 *  passing the operand as the parameter.
 */
template<typename A, typename Op, int N>
class ExprUnopNode: public LTLIterator
{
   private:
      A iter_;  //!< The operand of the unary operation

   public:
      //! the result data type is the \c value_type of the operation
      typedef typename Op::value_type value_type;

      //@{
      //! The number of dimensions
      enum { dims=N };
      //! The number of ltl::IndexIter index iterators in the operand
      enum { numIndexIter = A::numIndexIter };
      //! The number of convolution operations
      enum { numConvolution = A::numConvolution };
      //! Vectorizable if the operand is vectorizable and if the the operation is vectorizable
      enum { isVectorizable = A::isVectorizable * Op::isVectorizable };
      //@}

      //! Constructor
      inline ExprUnopNode( const A& a )
            : iter_(a)
      { }

      //@{
      //! Implement the iterator interface forwarding all operations to the operand
      inline void operator++()
      {
         ++iter_;
      }

      inline void advance()
      {
         iter_.advance();
      }

      inline void advance( const int i )
      {
         iter_.advance(i);
      }

      inline void advance( const int i, const int dim )
      {
         iter_.advance(i,dim);
      }

      inline void advanceWithStride1()
      {
         iter_.advanceWithStride1();
      }

      inline void advanceDim()
      {
         iter_.advanceDim();
      }

      inline void advanceDim(const int cutDim)
      {
         iter_.advanceDim(cutDim);
      }

      //@{
      //! Evaluate by passing the values of the operand to the operation
      inline value_type operator*() const
      {
         return Op::eval( *iter_ );
      }

      inline value_type readWithoutStride( const int i ) const
      {
         return Op::eval( iter_.readWithoutStride(i) );
      }

      inline value_type readWithStride( const int i ) const
      {
         return Op::eval( iter_.readWithStride(i) );
      }
      inline value_type readWithStride( const int i, const int dim ) const
      {
         return Op::eval( iter_.readWithStride(i,dim) );
      }
      //@}
      /*! compute and return the value of the subexpression in the parse tree below us
       *  at an offset in dimension 1, 2, and/or 3. Used for evaluating convolutions
       */
      value_type readAtOffsetDim( const int i, const int dim ) const
      {
         return Op::eval(iter_.readAtOffsetDim(i,dim));
      }
      value_type readAtOffset( const int i ) const
      {
         return Op::eval(iter_.readAtOffset(i));
      }
      value_type readAtOffset( const int i, const int j ) const
      {
         return Op::eval(iter_.readAtOffset(i,j));
      }
      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         return Op::eval(iter_.readAtOffset(i,j,k));
      }
      //@}

      //@{
      /*!
       *  Return the width of the lower/upper boundary in dimension \c dim to be left unevaluated.
       *  Zero in most cases except for convolutions, which do require padding of the half-width
       *  of the kernel.
       */
      int boundary_l(const int dim) const
      {
         return iter_.boundary_l(dim);
      }
      int boundary_u(const int dim) const
      {
         return iter_.boundary_u(dim);
      }
      //@}

#ifdef LTL_USE_SIMD
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(vec_value_type) readVec( const int i ) const
      {
         return Op::eval_vec( iter_.readVec(i) );
      }

      inline void alignData( const int align )
      {
         iter_.alignData( align );
      }

      inline int getAlignment() const
      {
         return _expr_getalign( iter_ );
      }

      inline bool sameAlignmentAs( const int p ) const
      {
         return iter_.sameAlignmentAs(p);
      }
#endif

      inline bool isStorageContiguous() const
      {
         return iter_.isStorageContiguous();
      }

      inline bool isStride1() const
      {
         return iter_.isStride1();
      }

      inline bool isConformable( const Shape<N>& other ) const
      {
         return iter_.isConformable( other );
      }

      inline void reset()
      {
         iter_.reset();
      }

      //! Return a \c shape object from the operand parse tree
      inline const Shape<N> *shape() const
      {
         return iter_.shape();
      }
      //@}
};


//! Node in the expression parse tree representing a literal number.
/*!
 *  \ingroup marray_expr
 *  This class represents a literal number in the parse tree of an expression so that
 *  expressions can involve literal constants in the usual way.
 *
 *  Like all parse tree elements, this class implements the iterator interface, simply
 *  doing nothing for all methods except dereferencing, for which it returns the literal
 *  constant it holds.
 */
template<typename T>
class ExprLiteralNode: public LTLIterator
{
   private:
#ifdef LTL_USE_SIMD
     T_VEC_TYPE(T) vec_;     //!< The vector literal constant
#endif
     const T f_;             //!< The literal constant

   public:
      //! the result data type is the type of the constant
      typedef T value_type;
      //@{
      //! The number of dimensions
      enum { dims=0 };
      //! The number of ltl::IndexIter index iterators (always 0)
      enum { numIndexIter = 0 };
      //! The number of convolution operations (always 0)
      enum { numConvolution = 0 };
      //! Constants are vectorizable: splat the constant across the vector
      enum { isVectorizable = 1 };
      //@}

      //! Constructor: store the constant, splat if we are vectorizing
      inline ExprLiteralNode( const T f )
      :
#ifdef LTL_USE_SIMD
         vec_(VEC_INIT(T,f)),
#endif
      f_(f)
      {  }

      //@{
      //! Implement the iterator interface: do nothing for advancing
      inline void advance() const
         { }

      inline void advance( const int ) const
         { }

      inline void advance( const int, const int ) const
         { }

      inline void advanceWithStride1() const
         { }

      inline void advanceDim() const
         { }

      inline void advanceDim(const int) const
         { }

      inline void operator++() const
         { }

      //@{
      //! Evaluate by simply returning the constant
      inline value_type operator*() const
      {
         return f_;
      }

      inline value_type readWithoutStride( const int ) const
      {
         return f_;
      }

      inline value_type readWithStride( const int ) const
      {
         return f_;
      }

      inline value_type readWithStride( const int, const int ) const
      {
         return f_;
      }
      //@}
      /*! compute and return the value of the subexpression in the parse tree below us
       *  at an offset in dimension 1, 2, and/or 3. Used for evaluating convolutions
       */
      value_type readAtOffsetDim( const int, const int ) const
      {
         return f_;
      }
      value_type readAtOffset( const int ) const
      {
         return f_;
      }
      value_type readAtOffset( const int, const int ) const
      {
         return f_;
      }
      value_type readAtOffset( const int, const int, const int ) const
      {
         return f_;
      }
      //@}

      //@{
      int boundary_l(const int) const
      {
         return 0;
      }
      int boundary_u(const int) const
      {
         return 0;
      }
      //@}

#ifdef LTL_USE_SIMD
      typedef T vec_value_type;
      inline typename VEC_TYPE(vec_value_type) readVec( const int) const
      {
         return vec_;
      }

      inline void alignData( const int /*align*/ )
         {  }

      inline int getAlignment() const
      {
         return -1;
      }

      inline bool sameAlignmentAs( const int ) const
      {
         return true;
      }
#endif

      //! always true
      inline bool isStorageContiguous() const
      {
         return true;
      }

      //! always true
      inline bool isStride1() const
      {
         return true;
      }

      //! always true
      template<int N>
      inline bool isConformable( const Shape<N>& ) const
      {
         return true;
      }

      inline void reset()
      {  }

      //@}
};



/// \cond DOXYGEN_IGNORE
//
// we need a trait class to distinguish leaf nodes that carry real
// iterators from node types that represent sub-expressions. Also,
// we have to deal with literals in expressions.
//
template<typename T>
struct ExprNodeType
{
   typedef ExprLiteralNode<T> expr_type;
   typedef typename expr_type::value_type value_type;
   static expr_type node(const T& x)
   {
      return expr_type(x);
   }
};

// Already an expression template term
template<typename T, int N>
struct ExprNodeType< ExprNode<T,N> >
{
   typedef ExprNode<T,N> expr_type;
   typedef typename expr_type::value_type value_type;
   static const expr_type node(const ExprBase<ExprNode<T,N>,N >& x)
   {
      return x.derived();
   }
};

// An array operand - here we need to call \c begin() to obtain an iterator
template<typename T, int N>
struct ExprNodeType< MArray<T,N> >
{
   typedef typename MArray<T,N>::ConstIterator expr_type;
   typedef T value_type;
   static expr_type node(const ExprBase<MArray<T,N>,N >& x)
   {
      return x.derived().begin();
   }
};/// \endcond



/*! \brief Define the global binary functions/operators for ltl::MArray expressions, version for 2 MArray operands, overloaded versions below.
 *  \ingroup marray_expr
 *  \hideinitializer
 *
 *  Each binary function/operator takes opjects derived from \c ExprBase
 *  (\c MArray, \c ExprNode), or literals as arguments and returns a parse-tree
 *  node for the operation it represents:
 *  \code
 *    ExprNode <ExprBinopNode <A, B, Operation, NDim> > function( A& rhs, B& lhs )
 *  \endcode
 *  where \c LHS and \c RHS are of type \c ExprBase or (scalar) literals.
 *
 *  There are 8 combination of argument types for binary ops, namely:
 *
 *    array   op  array,    \n
 *    array   op  scalar,   \n
 *    scalar  op  array,    \n
 *    expr    op  array,    \n
 *    array   op  expr,     \n
 *    expr    op  expr,     \n
 *    scalar  op  expr, and \n
 *    expr    op  scalar.   \n
 *
 *  An overloaded function/operator template is generated by these macros
 *  for each of these cases. The literal type is assumed to be of the same type
 *  as the elements of the expr (or be type-castable to that type).
 *
 *  This might seem like it could be reduced to 3 cases by unifying the
 *  overloaded functions for array and expr to take \c ExprBase objects
 *  as arguments. However, this causes ambiguous overloads with the cases
 *  taking an arbitrary type (a scalar).
 *
 */
#define BINOP_AA(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<typename MArray<T1,N>::ConstIterator,                 \
                              typename MArray<T2,N>::ConstIterator,                 \
                              op<T1, T2>,                                           \
                              N>,                                                   \
                   N>                                                               \
operator(const MArray<T1,N>& a, const MArray<T2,N>& b)                              \
{                                                                                   \
   typedef ExprBinopNode<typename MArray<T1,N>::ConstIterator,                      \
                         typename MArray<T2,N>::ConstIterator,                      \
                         op<T1, T2>,                                                \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a.begin(), b.begin()) );                         \
}

#define BINOP_AE(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<typename MArray<T1,N>::ConstIterator,                 \
                              ExprNode<T2,N>,                                       \
                              op<T1, typename T2::value_type>,                      \
                              N>,                                                   \
                   N>                                                               \
operator(const MArray<T1,N>& a, const ExprNode<T2,N>& b)                            \
{                                                                                   \
   typedef ExprBinopNode<typename MArray<T1,N>::ConstIterator,                      \
                         ExprNode<T2,N>,                                            \
                         op<T1, typename T2::value_type>,                           \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a.begin(), b) );                                 \
}

#define BINOP_EA(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<ExprNode<T1,N>,                                       \
                              typename MArray<T2,N>::ConstIterator,                 \
                              op<typename T1::value_type, T2>,                      \
                              N>,                                                   \
                   N>                                                               \
operator(const ExprNode<T1,N>& a, const MArray<T2,N>& b)                            \
{                                                                                   \
   typedef ExprBinopNode<ExprNode<T1,N>,                                            \
                         typename MArray<T2,N>::ConstIterator,                      \
                         op<typename T1::value_type, T2>,                           \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a, b.begin()) );                                 \
}

#define BINOP_AL(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<typename MArray<T1,N>::ConstIterator,                 \
                              ExprLiteralNode<T2>,                                  \
                              op<T1, T2>,                                           \
                              N>,                                                   \
                   N>                                                               \
operator(const MArray<T1,N>& a, const T2& b)                                        \
{                                                                                   \
   typedef ExprBinopNode<typename MArray<T1,N>::ConstIterator,                      \
                         ExprLiteralNode<T2>,                                       \
                         op<T1, T2>,                                                \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a.begin(), ExprLiteralNode<T2>(b)) );            \
}

#define BINOP_LA(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<ExprLiteralNode<T1>,                                  \
                              typename MArray<T2,N>::ConstIterator,                 \
                              op<T1, T2>,                                           \
                              N>,                                                   \
                   N>                                                               \
operator(const T1& a, const MArray<T2,N>& b)                                        \
{                                                                                   \
   typedef ExprBinopNode<ExprLiteralNode<T1>,                                       \
                         typename MArray<T2,N>::ConstIterator,                      \
                         op<T1, T2>,                                                \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(ExprLiteralNode<T1>(a), b.begin()) );            \
}

#define BINOP_EE(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<ExprNode<T1,N>,                                       \
                              ExprNode<T2,N>,                                       \
                              op<typename T1::value_type, typename T2::value_type>, \
                              N>,                                                   \
                   N>                                                               \
operator(const ExprNode<T1,N>& a, const ExprNode<T2,N>& b)                          \
{                                                                                   \
   typedef ExprBinopNode<ExprNode<T1,N>,                                            \
                         ExprNode<T2,N>,                                            \
                         op<typename T1::value_type, typename T2::value_type>,      \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a, b) );                                         \
}

#define BINOP_EL(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<ExprNode<T1,N>,                                       \
                              ExprLiteralNode<T2>,                                  \
                              op<typename T1::value_type, T2>,                      \
                              N>,                                                   \
                   N>                                                               \
operator(const ExprNode<T1,N>& a, const T2& b)                                      \
{                                                                                   \
   typedef ExprBinopNode<ExprNode<T1,N>,                                            \
                         ExprLiteralNode<T2>,                                       \
                         op<typename T1::value_type, T2>,                           \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(a, ExprLiteralNode<T2>(b)) );                    \
}

#define BINOP_LE(operator, op)                                                      \
template<typename T1, typename T2, int N>                                           \
inline ExprNode<ExprBinopNode<ExprLiteralNode<T1>,                                  \
                              ExprNode<T2,N>,                                       \
                              op<T1, typename T2::value_type>,                      \
                              N>,                                                   \
                   N>                                                               \
operator(const T1& a, const ExprNode<T2,N>& b)                                      \
{                                                                                   \
   typedef ExprBinopNode<ExprLiteralNode<T1>,                                       \
                         ExprNode<T2,N>,                                            \
                         op<T1, typename T2::value_type>,                           \
                         N>                                                         \
     ExprT;                                                                         \
   return ExprNode<ExprT,N>( ExprT(ExprLiteralNode<T1>(a), b) );                    \
}

/*! \brief Define the global unary operators, overloaded versions for marray operand.
 *  \ingroup marray_expr
 *  \hideinitializer
 *
 *  Each unary function/operator takes objects derived from \c ExprBase
 *   (\c ltl::MArray, \c ltl::ExprNode) as arguments
 *  and returns a parse-tree node for the operation it represents:
 *  \code
 *    ExprNode <ExprUnopNode <A, Operation, NDim> > function( A& operand )
 *  \endcode
 *  where \c operand is of type \c ExprBase.
 */
#define UNOP_E(operator, op)                                               \
template<typename T, int N>                                                \
inline ExprNode<ExprUnopNode<typename ExprNodeType<T>::expr_type,          \
                             op<typename T::value_type>,                   \
                             N>,                                           \
                N>                                                         \
operator(const ExprBase<T,N>& a )                                          \
{                                                                          \
   typedef ExprUnopNode<typename ExprNodeType<T>::expr_type,               \
                        op<typename T::value_type>,                        \
                        N>                                                 \
     ExprT;                                                                \
   return ExprNode<ExprT,N>( ExprT(ExprNodeType<T>::node(a.derived())) );  \
}

//@{
/*! \brief Make a binary (built-in) operator available to expression templates.
 *  \hideinitializer \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given binary operator \c operation. The return type is the
 *  standard C type-promoted result of the operation on built in scalar types.
 *
 *  It is assumed that the name of the applicative template for the same operator
 *  is called \c ltl::__ltl_opname and that this template is defined elsewhere
 *  (\ref misc/applicops.h for built-in operators).
 */
#define DECLARE_BINOP(operation, opname)                           \
      BINOP_AA(operator  operation,__ltl_##opname)                 \
      BINOP_AE(operator  operation,__ltl_##opname)                 \
      BINOP_EA(operator  operation,__ltl_##opname)                 \
      BINOP_AL(operator  operation,__ltl_##opname)                 \
      BINOP_LA(operator  operation,__ltl_##opname)                 \
      BINOP_EE(operator  operation,__ltl_##opname)                 \
      BINOP_EL(operator  operation,__ltl_##opname)                 \
      BINOP_LE(operator  operation,__ltl_##opname)
//@}

/*! \brief Make a unary (built-in) operator available to expression templates.
 *  \hideinitializer \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given unary operator \c operation. The return type is the
 *  standard C type-promoted result of the operation on built in scalar types.
 *
 *  It is assumed that the name of the applicative template for the same operator
 *  is called \c ltl::__ltl_opname and that this template is defined elsewhere
 *  (\ref misc/applicops.h for built-in operators).
 */
#define DECLARE_UNOP(operation, opname)         \
   UNOP_E(operator  operation,__ltl_##opname)


//@{
/*! \brief Make any binary function available to expression templates.
 *  \hideinitializer \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given binary function \c function. The return type may be different
 *  than the operand type.
 *
 *  It is assumed that the name of the applicative template for the same function
 *  is called \c ltl::__ltl_function and that this template is defined elsewhere.
 *  (\ref misc/applicops.h for standard functions).
 *
 *  The function itself has to be implemented with the signature
 *  \code
 *    template <typename T>
 *    T function( const T& a, const T& b );
 *  \endcode
 */
#define DECLARE_BINARY_FUNC_(function)                      \
      BINOP_AA(function,__ltl_##function)                   \
      BINOP_AE(function,__ltl_##function)                   \
      BINOP_EA(function,__ltl_##function)                   \
      BINOP_AL(function,__ltl_##function)                   \
      BINOP_LA(function,__ltl_##function)                   \
      BINOP_EE(function,__ltl_##function)                   \
      BINOP_EL(function,__ltl_##function)                   \
      BINOP_LE(function,__ltl_##function)
//@}


/*! \brief Make any unary function available to expression templates.
 *  \hideinitializer \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given unary function \c function. The return type may be different
 *  than the operand type.
 *
 *  It is assumed that the name of the applicative template for the same function
 *  is called \c ltl::__ltl_function and that this template is defined elsewhere
 *  (\ref misc/applicops.h for standard functions).
 *
 *  The function itself has to be implemented with the signature
 *  \code
 *    template <typename T>
 *    T function( const T& a );
 *  \endcode
 */
#define DECLARE_UNARY_FUNC_(function) \
     UNOP_E(function,__ltl_##function)


/*! \brief Make any user-defined binary function available to expression templates.
 *  \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given binary function \c function. The return type may be different
 *  than the operand type.
 *
 *  This macro also declares and defines the applicative templates for this function.
 *  It is the only macro that needs to be called by users to make user-defined functions
 *  available to expression templates.
 *
 *  Assume you have a function like this:
 *  \code
 *    template <typename T>
 *    ret_type function( const T& a, const T& b );
 *  \endcode
 *  Then using
 *  \code
 *    DECLARE_BINARY_FUNC(function, ret_type);
 *  \endcode
 *  This function will be usable in expression templates.
 */
#define DECLARE_BINARY_FUNC(function, ret_type)              \
MAKE_BINAP_FUNC( __ltl_##function, ret_type, function );     \
DECLARE_BINARY_FUNC_(function)


/*! \brief Make any user-defined unary function available to expression templates.
 * \ingroup marray_expr
 *
 *  This macro declares all necessary overloaded operators to build the parse tree
 *  for a given unary function \c function. The return type may be different
 *  than the operand type.
 *
 *  This macro also declares and defines the applicative templates for this function.
 *  It is the only macro that needs to be called by users to make user-defined functions
 *  available to expression templates.
 *
 *  Assume you have a function like this:
 *  \code
 *    template <typename T>
 *    ret_type function( const T& a );
 *  \endcode
 *  Then using
 *  \code
 *    DECLARE_BINARY_FUNC(function, ret_type);
 *  \endcode
 *  This function will be usable in expression templates.
 */
#define DECLARE_UNARY_FUNC(function, ret_type)                \
MAKE_UNAP_FUNC( __ltl_##function, ret_type, function );       \
DECLARE_UNARY_FUNC_(function)

#ifdef LTL_USE_SIMD
// The same convenience macros for vectorized functions
//
#define DECLARE_UNARY_FUNC_VEC(function, ret_type, vec_impl) \
MAKE_UNAP_FUNC_VEC(__ltl_##function, ret_type, function, vec_impl)

#define DECLARE_BINARY_FUNC_VEC(function, ret_type, vec_impl) \
MAKE_BINAP_FUNC_VEC( __ltl_##function, ret_type, function, vec_impl )
#endif

}

#endif

