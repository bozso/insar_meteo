/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: convolve.h 551 2015-02-03 16:04:19Z drory $
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

#ifndef __LTL_MARRAY_CONVOLVE_H__
#define __LTL_MARRAY_CONVOLVE_H__

#include <ltl/fvector.h>

namespace ltl {

/*! \file convolve.h
    \ingroup kernels

  Convolution kernels for \c MArrays implementing various smoothing
  kernels and finite differences methods that can be used in any \c MArray
  expression. Currently, convoltions are implemented for expressions of up
  to rank 3.
*/

/*! Convolution support and kernels in MArray expressions

  \defgroup kernels Convolution support and kernels in MArray expressions
  \ingroup marray

Convolution kernels for \c MArrays implementing various smoothing
kernels and finite differences methods that can be used in any \c MArray
expression. Currently, convoltions are implemented for expressions of up
to rank 3.

Convolution operations are used by calling
\code
   convolve( Expr, Kernel )
\endcode

within any \c MArray expression. \c Expr itself may be any valid \c MArray expression
(partial reductions very likely do not produce correct results right now, though).

<b>Declaring and implementing a convolution kernel.</b>

As an example of a declaration of a convolution kernel, let us consider the
o(h^2) first derivative along the dimension dim using central differences:

\code
DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv1,A)
    return OFFSET1(A,1,dim) - OFFSET1(A,-1,dim);
END_CONVOLUTION_OPERATOR;
\endcode

The macros \c OFFSET0(), \c OFFSET(i,dim), \c OFFSET1(i), \c OFFSET2(i,j), \c OFFSET3(i,j,k)
allow access to elements of an expression relative to the current position.
The offsets can be along any one particular dimension, or along dimensions one, two, and/or three (currently,
convolutions are only supported in expression up to rank 3). \c OFFSET0() is
short for the current element (element at offset 0).

The extent of the convolution kernel (the maximum offsets used in the
kernel) are determined automatically for you before the operation is executed.
The extent can be different in each dimension and in each direction within a
dimension. The convolution will only be evaluated for elements of the operand expression
for which the kernel lies completely within the domain of the expression. <b>The elements
at the boundary will not be accessed.</b>

This kernel can then be used in expressions:

\code
    MArray<float,2> B, C;
    ...
    B = 1/(2*h) * convolve( C*C/2, cderiv1(1) );
\endcode

Which would assign the first derivative along the first dimension of \c C*C/2 to \c B. Note
the parameter in the constructor of the kernel \c cderiv1(1). This is the dimension along which
we want to take the derivative.

The \c DECLARE_CONVOLUTION_OPERATOR_DIM macro creates a kernel with one integer member, the one
storing the dimension that is given in the constructor call above. If no argument is needed,
the \c DECLARE_CONVOLUTION_OPERATOR version of the macro should be used. In that case, nothing
is stored within the kernel struct.

Various finite-differences kernels are provided by ltl out of the box (first to fourth derivatives
based on central, forward, or backward differences, Laplacians, ...). For a complete list and
some more documentation, see the file \ref convolve.h.

Note that convolutions can provide higher-order tensors than their operands. An example
is the gradient operator (there are no convenience macros yet, so definition has to be
provided by hand:

\code
struct grad3D
{
      template<typename Iter>
      inline FVector<typename Iter::value_type,3> eval(Iter& A) const
      {
         FVector<typename Iter::value_type,3> g;
         g(1) = OFFSET(A,1,1) - OFFSET(A,-1,1);
         g(2) = OFFSET(A,1,2) - OFFSET(A,-1,2);
         g(3) = OFFSET(A,1,3) - OFFSET(A,-1,3);
         return g;
      }
};
template <typename Iter>
struct kernel_return_type<Iter,grad3D>
{
   typedef FVector<typename Iter::value_type,3>  value_type;
};
\endcode

Note that the \c kernel_return_type trait has to be in namespace \c ltl.

With this definition of the 3-D gradient operator we can write
\code
MArray<float,2> A(10,10);
MArray<FVector<float,2>,2> B(10,10);
A = 0.0f;
A(5,5) = 1.0f;
B = convolve(A,grad2D());
\endcode

As a more realistic example for the use of these kernels, consider solving
a single timestep in the acoustic wave propagation equation in 3D.
Let P_tm1, P_t, and P_tp1 be the pressure field
at times tm1=t-1, t, and tp1=t+1, and c be the speed of sound.

Without convolution expressions, this would be implemented like this:

\code
   Range X(2,N-1), Y(2,N-1), Z(2,N-1);

   P_tp1(X,Y,Z) = (2-6*c(X,Y,Z)) * P_t(X,Y,Z)
                + c(X,Y,Z)*(P_t(X-1,Y,Z) + P_t(X+1,Y,Z) + P_t(X,Y-1,Z) + P_t(X,Y+1,Z) + P_t(X,Y,Z-1) + P_t(X,Y,Z+1))
                - P_tm1(X,Y,Z);
\endcode

With convolution expressions, this becomes:

\code
  P_tp1 = 2 * P_t + c * convolve(P_t, Laplacian3D) - P_tm1;
\endcode

Of course, standard convolution kernels can be implemented in N-Dimensions as well.
As an example, here's a 1-D Gaussian:

\code
struct GaussianKernel1D
{
      GaussianKernel1D (const double s, const double k) :
            sigma_ (s), extent_ (k)
      {
         norm_ = 0.0;
         for (int i = -extent_; i<=extent_; ++i)
            norm_ += exp (-0.5*(double (i*i))/(sigma_*sigma_));
      }

      template<typename Iter>
      inline typename Iter::value_type eval (Iter& A) const
      {
         double r = 0.0;
         for (int i = -extent_; i<=extent_; ++i)
            r += exp (-0.5*(double (i*i))/(sigma_*sigma_))*OFFSET1(A, i);

         return typename Iter::value_type (r/norm_);
      }

      double sigma_, norm_;
      int extent_;
};

template <typename Iter>
struct kernel_return_type<Iter,GaussianKernel1D>
{
   typedef typename Iter::value_type  value_type;
};
\endcode

*/


//! access the element at the current position (offset 0)
#define OFFSET0(A)       (*A)

//! access the element at offset i in the dimension dim
#define OFFSET(A,i,dim)  (A.readAtOffsetDim(i,dim))

//! access the element at offset i in the first dimension (the one with the smallest stride).
#define OFFSET1(A,i)     (A.readAtOffset(i))

//! access the element at offset i in the first, and j in the second dimension.
#define OFFSET2(A,i,j)   (A.readAtOffset(i,j))

//! access the element at offset i in the first, j in the second, and k in the third dimension.
#define OFFSET3(A,i,j,k) (A.readAtOffset(i,j,k))

/*! declare and implement a convolution kernel that needs no parameters.
    The macro argument \c name
    will be used to call the kernel using \c convolve(Expr, kernel()).
    The parameter \c A is used in the implementation to refer to the
    expression to convolve at runtime.

    It is assumed that the convolution operation yields the same
    data type as the operand of the convolution.

  \code
  DECLARE_CONVOLUTION_OPERATOR(mykernel,A)
      return OFFSET1(A,1) - OFFSET1(A,-1) ... ;
  END_CONVOLUTION_OPERATOR;
  \endcode
*/
#define DECLARE_CONVOLUTION_OPERATOR(name,A)                 \
struct name                                                  \
{                                                            \
      template<typename Iter>                                \
      static inline typename Iter::value_type eval(Iter& A)  \
      {


/*! declare and implement a convolution kernel that takes one
    integer argument.
    The macro argument \c name
    will be used to call the kernel using \c convolve(Expr, kernel(int)).
    The parameter \c A is used in the implementation to refer to the
    expression to convolve at runtime. The int that is passed
    in the constructor may be used inside the evaluation function.
    It is meant to specify a dimension, such as in the
    \c cderiv(int dim) kernels (derivatives along dimension dim).

    It is assumed that the convolution operation yields the same
    data type as the operand of the convolution.

  \code
  // first derivative using central differences along
  // dimension dim:
  DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv1,A)
      return OFFSET(A,1,dim) - OFFSET(A,-1,dim);
  END_CONVOLUTION_OPERATOR;
  \endcode
*/
#define DECLARE_CONVOLUTION_OPERATOR_DIM(name,A)             \
struct name                                                  \
{                                                            \
      name( const int d ) : dim(d) {}                        \
      const int dim;                                         \
      template<typename Iter>                                \
      inline typename Iter::value_type eval(Iter& A) const   \
      {

/// \cond DOXYGEN_IGNORE
#define END_CONVOLUTION_OPERATOR                             \
      }                                                      \
}
/// \endcond


template <typename Iter, typename Kernel>
struct kernel_return_type
{
   typedef typename Iter::value_type value_type;
};


//@{
/*!
 * first to fourth derivative along dimension dim, central differences, second-order accurate,
 * (multiply with factors 2h, h^2, 2h^3, h^4 for non-unity h = delta x)
 */
DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv1,A)
  return OFFSET(A,1,dim) - OFFSET(A,-1,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv2,A)
  return -2.0 * (*A) + OFFSET(A,1,dim) + OFFSET(A,-1,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv3,A)
  return -2.0 * (OFFSET(A,1,dim) - OFFSET(A,-1,dim)) + (OFFSET(A,2,dim) - OFFSET(A,-2,dim));
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(cderiv4,A)
  return 6.0 * (*A) - 4.0 * (OFFSET(A,1,dim) + OFFSET(A,-1,dim)) + (OFFSET(A,2,dim) + OFFSET(A,-2,dim));
END_CONVOLUTION_OPERATOR;
//@}

/****************************************************************************
 * Forward differences with accuracy O(h)
 ****************************************************************************/

//@{
/*!
 * first to fourth derivative along dimension dim, forward differences, first-order accurate,
 */
DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv1h,A)
  return -(*A) + OFFSET(A,1,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv2h,A)
  return (*A) - 2.0 * OFFSET(A,1,dim) + OFFSET(A,2,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv3h,A)
  return -(*A) + 3.0 * OFFSET(A,1,dim) - 3.0 * OFFSET(A,2,dim) + OFFSET(A,3,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv4h,A)
  return (*A) - 4.0 * OFFSET(A,1,dim) + 6.0 * OFFSET(A,2,dim)
    - 4.0 * OFFSET(A,3,dim) + OFFSET(A,4,dim);
END_CONVOLUTION_OPERATOR;
//@}

//@{
/*!
 * first to fourth derivative along dimension dim, forward differences, second-order accurate,
 */
DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv1h2,A)
  return -3.0 * (*A) + 4.0 * OFFSET(A,1,dim) - OFFSET(A,2,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv2h2,A)
  return 2.0 * (*A) - 5.0 * OFFSET(A,1,dim) + 4.0 * OFFSET(A,2,dim)
    - OFFSET(A,3,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv3h2,A)
  return -5.0 * (*A) + 18.0 * OFFSET(A,1,dim) - 24.0 * OFFSET(A,2,dim)
    + 14.0 * OFFSET(A,3,dim) - 3.0 * OFFSET(A,4,dim);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR_DIM(fderiv4h2,A)
  return 3.0 * (*A) - 14.0 * OFFSET(A,1,dim) + 26.0 * OFFSET(A,2,dim)
    - 24.0 * OFFSET(A,3,dim) + 11.0 * OFFSET(A,4,dim) - 2.0 * OFFSET(A,5,dim);
END_CONVOLUTION_OPERATOR;
//@}

//@{
/*!
 * Laplacian in 2D and 3D to o(h^2) and o(h^4).
 *
 */
DECLARE_CONVOLUTION_OPERATOR(Laplacian2D,A)
  return -4.0 * (*A) + OFFSET2(A,-1,0) + OFFSET2(A,1,0) + OFFSET2(A,-1,1) + OFFSET2(A,1,1);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR(Laplacian2D4,A)
  return -60.0 * (*A)
         + 16.0 * (OFFSET2(A,-1,0) + OFFSET2(A,1,0) + OFFSET2(A,-1,1) + OFFSET2(A,1,1))
         -        (OFFSET2(A,-2,0) + OFFSET2(A,2,0) + OFFSET2(A,-2,1) + OFFSET2(A,2,1));
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR(Laplacian3D,A)
  return -6.0 * (*A)
         + OFFSET2(A,-1,0) + OFFSET2(A,1,0)
         + OFFSET2(A,-1,1) + OFFSET2(A,1,1)
         + OFFSET2(A,-1,2) + OFFSET2(A,1,2);
END_CONVOLUTION_OPERATOR;

DECLARE_CONVOLUTION_OPERATOR(Laplacian3D4,A)
  return -90.0 * (*A)
         + 16.0 * (OFFSET2(A,-1,0) + OFFSET2(A,1,0) + OFFSET2(A,-1,1) + OFFSET2(A,1,1) + OFFSET2(A,-1,2) + OFFSET2(A,1,2))
         -        (OFFSET2(A,-2,0) + OFFSET2(A,2,0) + OFFSET2(A,-2,1) + OFFSET2(A,2,1) + OFFSET2(A,-2,2) + OFFSET2(A,2,2));
END_CONVOLUTION_OPERATOR;
//@}

//@{
/*!
 * Gradient operators in 2D and 3D
 *
 */
struct grad2D
{
      template<typename Iter>
      inline FVector<typename Iter::value_type,2> eval(Iter& A) const
      {
         FVector<typename Iter::value_type,2> g;
         g(1) = OFFSET(A,1,1) - OFFSET(A,-1,1);
         g(2) = OFFSET(A,1,2) - OFFSET(A,-1,2);
         return g;
      }
};
template <typename Iter>
struct kernel_return_type<Iter,grad2D>
{
   typedef FVector<typename Iter::value_type,2>  value_type;
};

struct grad3D
{
      template<typename Iter>
      inline FVector<typename Iter::value_type,3> eval(Iter& A) const
      {
         FVector<typename Iter::value_type,3> g;
         g(1) = OFFSET(A,1,1) - OFFSET(A,-1,1);
         g(2) = OFFSET(A,1,2) - OFFSET(A,-1,2);
         g(3) = OFFSET(A,1,3) - OFFSET(A,-1,3);
         return g;
      }
};
template <typename Iter>
struct kernel_return_type<Iter,grad3D>
{
   typedef FVector<typename Iter::value_type,3>  value_type;
};
//@}


/// \cond DOXYGEN_IGNORE
/*!
 *  extent calculator. mimics the convolution expression node, but only records
 *  the minimum and maximum offset read along each dimension instead of
 *  actually evaluating anything.
 *
 *  used by the convolution expr. node to automatically find out the extent
 *  of the kernel to avoid out of bounds access.
 */
template <int N>
struct extent_calculator
{
      typedef int value_type;

      FixedVector<int,N> extent_l;
      FixedVector<int,N> extent_u;

      extent_calculator()
      {
         extent_l=0;
         extent_u=0;
      }

      value_type operator*()
      {
         return readAtOffset(0);
      }

      value_type readAtOffsetDim( const int i, const int dim )
      {
         extent(dim,i);
         return 0;
      }

      value_type readAtOffset( const int i )
      {
         extent(1,i);
         return 0;
      }

      value_type readAtOffset( const int i, const int j )
      {
         extent(1,i);
         extent(2,j);
         return 0;
      }

      value_type readAtOffset( const int i, const int j, const int k )
      {
         extent(1,i);
         extent(2,j);
         extent(3,k);
         return 0;
      }

      void extent( const int dim, const int offset )
      {
         extent_l(dim) = std::min(extent_l(dim),offset);
         extent_u(dim) = std::max(extent_u(dim),offset);
      }
};

/*!
 *  Node type for Convolution Expressions
 *  Imnplements the iterator interface shared by all iterators and expression parse
 *  tree nodes.
 *
 *  It forwards all operations to the convolution operation. The special thing about
 *  evaluating convolution operatiions is that we pass the iterator to the kernel
 *  so that the kernel can access the expression at offset positions. Also,
 *
 *  The template parameters are an ExprBase to apply the convolution to
 *  and a Kernel.
 */
template<typename E, typename K, int N>
class ConvolveExpr : public ExprBase< ConvolveExpr<E,K,N>, N >, public LTLIterator
{
   public:
      //typedef typename E::value_type value_type;
      typedef typename kernel_return_type<E,K>::value_type value_type;
      enum { dims=N };
      enum { numIndexIter = E::numIndexIter };
      enum { numConvolution = E::numConvolution + 1};
      enum { isVectorizable = 0 };

      ConvolveExpr( const E& e, const K& k )
            : E_(e), K_(k)
      {
         LTL_ASSERT_((numConvolution)==1, "Cannot evaluate expression with nested convolution operations!");
         K_.eval(ec_);
      }

      ConvolveExpr( const ConvolveExpr<E,K,N>& other )
            :  E_(other.E_), K_(other.K_), ec_(other.ec_)
      { }

      value_type operator*() const
      {
         // call the Kernel's eval() passing the iterator-style object as a parameter
         return K_.eval(E_);
      }

      void operator++()
      {
         ++E_;
      }

      //! should never be called on \c ConvolveExpr, as we are evaluated
      // using only operator*()
      //@{
      value_type readWithoutStride( const int i ) const
      {
         LTL_ASSERT_(false, "Calling readWithoutStride on ConvolutionExpr!");
      }

      value_type readWithStride( const int i ) const
      {
         LTL_ASSERT_(false, "Calling readWithStride on ConvolutionExpr!");
      }

      value_type readWithStride( const int i, const int dim ) const
      {
         LTL_ASSERT_(false, "Calling readWithStride on ConvolutionExpr!");
      }

      value_type readAtOffsetDim( const int i, const int dim )
      {
         LTL_ASSERT_(false, "Calling readAtOffsetDim on ConvolutionExpr!");
      }

      value_type readAtOffset( const int i ) const
      {
         LTL_ASSERT_(false, "Calling readAtOffset on ConvolutionExpr!");
      }

      value_type readAtOffset( const int i, const int j ) const
      {
         LTL_ASSERT_(false, "Calling readAtOffset on ConvolutionExpr!");
      }

      value_type readAtOffset( const int i, const int j, const int k ) const
      {
         LTL_ASSERT_(false, "Calling readAtOffset on ConvolutionExpr!");
      }
      //@}

      int boundary_l(const int dim) const
      {
         return ec_.extent_l(dim);
      }

      int boundary_u(const int dim) const
      {
         return ec_.extent_u(dim);
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
      // Convolutions are not vectorizable
      typedef value_type vec_value_type;
      inline typename VEC_TYPE(value_type) readVec( const int i ) const
      { }

      inline void alignData( const int align )
      { }

      inline int getAlignment() const
      { }

      inline bool sameAlignmentAs( const int p ) const
      { }
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
      E E_;  //!< the expression operand
      K K_;  //!< the convolution Kernel
      extent_calculator<N> ec_;
};

/// \endcond

/*! global convolve() function template
 *  this template takes \c ExprBase objects and returns an \c ExprNode representing the
 *  convolution operation
*/
template<typename E, typename Kernel, int N>
inline ExprNode< ConvolveExpr<typename ExprNodeType<E>::expr_type, Kernel, N>, N>
convolve( const ExprBase<E,N>& A, const Kernel K )
{
   typedef ConvolveExpr<typename ExprNodeType<E>::expr_type,Kernel,N> ExprT;
   return ExprNode<ExprT,N>( ExprT( ExprNodeType<E>::node(A.derived()), K ) );
}

}

#endif // __LTL_MARRAY_CONVOLVE_H__
