/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: cast.h 491 2011-09-02 19:36:39Z drory $
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


#if !defined(__LTL_IN_FILE_MARRAY__)
#error "<ltl/marray/cast.h> must be included via ltl headers, never alone!"
#endif


#ifndef __LTL_MARRAY_CAST_H__
#define __LTL_MARRAY_CAST_H__


namespace ltl {

/*! \file cast.h

  Type cast operations for MArrays and expressions

  Use as \code cast<type>()( MArray ) \endcode or \code
  cast<type>()(expression) \endcode .
*/

/// \cond DOXYGEN_IGNORE
//{@
//! The applicative template to do the cast
template<typename To_type, typename From_type>
struct __ltl_cast : public _et_applic_base
{
   typedef To_type value_type;
   static inline To_type eval( const From_type& a )
      { return static_cast<To_type>(a); }
};
//@}
/// \endcond

/*! \defgroup cast Static type cast in MArray expressions.
 *
 *  \ingroup marray_class
 *
 *  Static type-cast operation for \c MArrays and Expressions.
 *
 *  Use as
 *  \code
 *    cast<type>()( MArray )
 *  \endcode
 *  or
 *  \code
 *    cast<type>()(expression)
 *  \endcode
 *  in any statement or expression involving ltl::MArray objects or expressions.
 *  This results in the value of the MArray or expression being \c static_casted
 *  to the type \c type while evaluating the expression.
 *
 */

//{@

/*!
 *  Type cast operation for \c MArray. Usage and semantics are
 *  (almost) as \c static_cast<T>() operation. For example, to use the
 *  elements of an \code MArray<float,2> A \endcode as \c ints, use
 *  \code cast<int>()( A ) \endcode . See the extra parenthesis to instantiate the \c
 *  cast instance?
 */
template<typename To_Type>
struct cast
{
   //! cast<> operation for \c ExprBase arguments
   template<typename T1, int N>
   inline ExprNode<ExprUnopNode<typename ExprNodeType<T1>::expr_type,
                                __ltl_cast<To_Type,typename T1::value_type>,
                                N >,
                   N>
   operator()( const ExprBase<T1,N>& a )
   {
      typedef ExprUnopNode<typename ExprNodeType<T1>::expr_type,
                           __ltl_cast<To_Type,typename T1::value_type>,
                           N >
         ExprT;
      return ExprNode<ExprT,N>( ExprT( ExprNodeType<T1>::node(a.derived()) ) );
   }
};
//@}



}


#endif // __LTL_MARRAY_CAST_H__
