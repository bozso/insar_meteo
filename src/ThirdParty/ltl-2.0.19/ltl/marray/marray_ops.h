/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marray_ops.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/marray/marray_ops.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_MARRAYOPS__
#define __LTL_MARRAYOPS__


namespace ltl {

// --------------------------------------------------------------------
/*! \file marray_ops.h
  `\name Overloaded X= operators. There is a version for an
  \code MArray rhs, an expression rhs, and a literal rhs for each operator.
  To have a single implementation of mathematical operations for scalar and
  vectorized code (where the C language extensions do not define X= ), we
  transform the \code A x= E \endcode assignment into \code A = A x E \endcode
  and call \code operator= \endcode .
*/
// --------------------------------------------------------------------


/*! \defgroup marray_assignment Assignment overloads

\ingroup marray_class

*/
//@{

//! \c operator+=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator+=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TAdd<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator+=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator+=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TAdd<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator+=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator+=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TAdd<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator-=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator-=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TSub<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator-=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator-=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TSub<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator-=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator-=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TSub<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator*=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator*=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TMul<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator*=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator*=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TMul<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator*=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator*=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TMul<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator/=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator/=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TDiv<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator/=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator/=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TDiv<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator/=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator/=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TDiv<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator|=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator|=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TBitOr<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator|=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator|=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TBitOr<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator|=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator|=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TBitOr<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator&=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator&=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TBitAnd<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator&=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator&=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TBitAnd<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator&=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator&=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TBitAnd<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator^=
template<typename T, int N>
template<typename T2>
inline MArray<T,N>& MArray<T,N>::operator^=( const MArray<T2,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      typename MArray<T2,N>::ConstIterator,
      __ltl_TBitXor<T ,T2>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a.begin())) );
}

//! \c operator^=
template<typename T, int N>
template<typename Expr>
inline MArray<T,N>& MArray<T,N>::operator^=( const ExprNode<Expr,N>& a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T ,N>::ConstIterator,
      ExprNode<Expr,N>,
      __ltl_TBitXor<T ,typename Expr::value_type>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//! \c operator^=
template<typename T, int N>
inline MArray<T,N>& MArray<T,N>::operator^=( const T a )
{
   LTL_ASSERT( memBlock_, "Assignment to uninitialized MArray!" );
   typedef ExprBinopNode<typename MArray<T,N>::ConstIterator,
      ExprLiteralNode<T>,
      __ltl_TBitXor<T,T>, N >
      ExprT;
   return this->operator=( ExprNode<ExprT,N>(ExprT(this->begin(),a)) );
}

//@}

}

#endif // __LTL_MARRAYOPS__
