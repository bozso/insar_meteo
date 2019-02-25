/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: expr_ops.h 527 2013-12-09 15:49:52Z cag $
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
#error "<ltl/marray/expr_ops.h> must be included via <ltl/marray.h>, never alone!"
#endif


#ifndef __LTL_EXPR_OPS__
#define __LTL_EXPR_OPS__

namespace ltl {

//
// now actually define operator template functions
// using the macros defined in <ltl/expr.h>
//

// standard operators
//
DECLARE_BINOP(+, TAdd)
DECLARE_BINOP(-, TSub)
DECLARE_BINOP(*, TMul)
DECLARE_BINOP(/, TDiv)
DECLARE_BINOP(&&, TAnd)
DECLARE_BINOP(||, TOr)

DECLARE_BINOP(& , TBitAnd )
DECLARE_BINOP(| , TBitOr )
DECLARE_BINOP(^ , TBitXor )
DECLARE_BINOP(% , TMod )

DECLARE_BINOP(> , TGT )
DECLARE_BINOP(< , TLT )
DECLARE_BINOP(>=, TGE )
DECLARE_BINOP(<=, TLE )
DECLARE_BINOP(!=, TNE )
DECLARE_BINOP(==, TEQ )

DECLARE_UNOP( +,  TPlus )
DECLARE_UNOP( -,  TMinus )
DECLARE_UNOP( !,  TNot )
DECLARE_UNOP( ~,  TNeg )

// Safe floating point comparison
DECLARE_BINARY_FUNC_(feq)
DECLARE_BINARY_FUNC_(fneq)


// standard library math functions
// (some will also be available on complex<T>)
//
DECLARE_UNARY_FUNC_( sin  )
DECLARE_UNARY_FUNC_( cos  )
DECLARE_UNARY_FUNC_( tan  )
DECLARE_UNARY_FUNC_( asin )
DECLARE_UNARY_FUNC_( acos )
DECLARE_UNARY_FUNC_( atan )
DECLARE_UNARY_FUNC_( sinh )
DECLARE_UNARY_FUNC_( cosh )
DECLARE_UNARY_FUNC_( tanh )
DECLARE_UNARY_FUNC_( exp  )
DECLARE_UNARY_FUNC_( log  )
DECLARE_UNARY_FUNC_( log10)
DECLARE_UNARY_FUNC_( sqrt )
DECLARE_UNARY_FUNC_( fabs )
DECLARE_UNARY_FUNC_( floor)
DECLARE_UNARY_FUNC_( ceil )
DECLARE_UNARY_FUNC_( round )

DECLARE_BINARY_FUNC_( pow  )
DECLARE_BINARY_FUNC_( fmod )
DECLARE_BINARY_FUNC_( atan2)


// IEEE math functions
//
#ifdef HAVE_IEEE_MATH
DECLARE_UNARY_FUNC_( asinh )
DECLARE_UNARY_FUNC_( acosh )
DECLARE_UNARY_FUNC_( atanh )
DECLARE_UNARY_FUNC_( cbrt  )
DECLARE_UNARY_FUNC_( expm1 )
DECLARE_UNARY_FUNC_( log1p )
DECLARE_UNARY_FUNC_( erf   )
DECLARE_UNARY_FUNC_( erfc  )
DECLARE_UNARY_FUNC_( j0    )
DECLARE_UNARY_FUNC_( j1    )
DECLARE_UNARY_FUNC_( y0    )
DECLARE_UNARY_FUNC_( y1    )
DECLARE_UNARY_FUNC_( lgamma)
DECLARE_UNARY_FUNC_( rint  )

DECLARE_BINARY_FUNC_( hypot )

DECLARE_UNARY_FUNC_( fpclassify )
DECLARE_UNARY_FUNC_( isfinite )
DECLARE_UNARY_FUNC_( isinf )
DECLARE_UNARY_FUNC_( isnan )
DECLARE_UNARY_FUNC_( isnormal )
#endif

// these are defined in misc/applicops.h
//
DECLARE_UNARY_FUNC_( pow2 )
DECLARE_UNARY_FUNC_( pow3 )
DECLARE_UNARY_FUNC_( pow4 )
DECLARE_UNARY_FUNC_( pow5 )
DECLARE_UNARY_FUNC_( pow6 )
DECLARE_UNARY_FUNC_( pow7 )
DECLARE_UNARY_FUNC_( pow8 )


#ifdef LTL_COMPLEX_MATH
// operations ONLY available on complex numbers
// argument complex<T>, but returning T
DECLARE_UNARY_FUNC_( abs)
DECLARE_UNARY_FUNC_( arg)
DECLARE_UNARY_FUNC_( norm )
DECLARE_UNARY_FUNC_( real )
DECLARE_UNARY_FUNC_( imag )
// argument complex<T> and returning complex<T>
DECLARE_UNARY_FUNC_( conj )
#endif
}

#endif
