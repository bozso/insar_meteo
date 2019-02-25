/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: expr_base.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/marray/expr_base.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_EXPR_BASE__
#define __LTL_EXPR_BASE__

/*! \file expr_base.h
 *  \brief This file defines the base class for all operands in
 *  expression templates.
 *
 *  \ingroup marray_expr
 */

namespace ltl {

/*!
 * \ingroup marray_expr
 *  Base class for all operands in expression templates. Both
 *  nodes and leafs of expressions inherit from this type.
 *
 *  This class uses the Curiously Recurring Template Pattern (CRTP) to
 *  implement "compile-time polymorphism" by taking the derived type as a
 *  template parameter and providing a method \c derived()
 *  to cast itself to the derived type.
 *
 *  The rank of the expression operand is made explicit through the
 *  second template parameter \c N_Dims to allow strict type checking
 *  on the rank in functions that accept \c ExprBase objects as parameters
 *  (e.g. make sure that parameters have the same rank, or some
 *  predetermined rank).
 */
template <typename Derived_T, int N_Dims>
struct ExprBase
{
   enum { dims = N_Dims };

   ExprBase()
   { }

   ExprBase(const ExprBase&)
   { }

   Derived_T& derived()
   {
      return static_cast<Derived_T&> (*this);
   }

   const Derived_T& derived() const
   {
      return static_cast<const Derived_T&> (*this);
   }
};

}

#endif //__LTL_EXPR_BASE__
