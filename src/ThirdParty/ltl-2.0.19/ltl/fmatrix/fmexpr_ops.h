/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmexpr_ops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FMATRIX__
#error "<ltl/fmatrix/fmexpr_ops.h> must be included via <ltl/fmatrix.h>, never alone!"
#endif


#ifndef __LTL_APPLICOPS_H__
#include <ltl/misc/applicops.h>
#endif

#ifndef __LTL_FMEXPR_OPS__
#define __LTL_FMEXPR_OPS__

namespace ltl
{

//
// now actually define operator template functions
// using the macros defined in <ltl/fvector/fvexpr.h>
//

// standard operators
//
DECLARE_FMBINOP(+, TAdd)

DECLARE_FMBINOP(-, TSub)
DECLARE_FMBINOP(*, TMul)
DECLARE_FMBINOP(/, TDiv)
DECLARE_FMBINOP(&&, TAnd)
DECLARE_FMBINOP(||, TOr)

DECLARE_FMBINOP(& , TBitAnd )
DECLARE_FMBINOP(| , TBitOr )
DECLARE_FMBINOP(^ , TBitXor )
DECLARE_FMBINOP(% , TMod )

DECLARE_FMBINOP(> , TGT )
DECLARE_FMBINOP(< , TLT )
DECLARE_FMBINOP(>=, TGE )
DECLARE_FMBINOP(<=, TLE )
DECLARE_FMBINOP(!=, TNE )
DECLARE_FMBINOP(==, TEQ )

DECLARE_FMUNOP( +,  TPlus )
DECLARE_FMUNOP( -,  TMinus )
DECLARE_FMUNOP( !,  TNot )
DECLARE_FMUNOP( ~,  TNeg )


// standard library math functions
//
DECLARE_FMUNARY_FUNC_( sin )
DECLARE_FMUNARY_FUNC_( cos )
DECLARE_FMUNARY_FUNC_( tan )
DECLARE_FMUNARY_FUNC_( asin )
DECLARE_FMUNARY_FUNC_( acos )
DECLARE_FMUNARY_FUNC_( atan )
DECLARE_FMUNARY_FUNC_( sinh )
DECLARE_FMUNARY_FUNC_( cosh )
DECLARE_FMUNARY_FUNC_( tanh )
DECLARE_FMUNARY_FUNC_( exp )
DECLARE_FMUNARY_FUNC_( log )
DECLARE_FMUNARY_FUNC_( log10 )
DECLARE_FMUNARY_FUNC_( sqrt )
DECLARE_FMUNARY_FUNC_( fabs )
DECLARE_FMUNARY_FUNC_( floor )
DECLARE_FMUNARY_FUNC_( ceil )

DECLARE_FMBINARY_FUNC_( pow )
DECLARE_FMBINARY_FUNC_( fmod )
DECLARE_FMBINARY_FUNC_( atan2 )


// IEEE math functions
//
#ifdef HAVE_IEEE_MATH

DECLARE_FMUNARY_FUNC_( asinh )
DECLARE_FMUNARY_FUNC_( acosh )
DECLARE_FMUNARY_FUNC_( atanh )
DECLARE_FMUNARY_FUNC_( cbrt )
DECLARE_FMUNARY_FUNC_( expm1 )
DECLARE_FMUNARY_FUNC_( log1p )
DECLARE_FMUNARY_FUNC_( erf )
DECLARE_FMUNARY_FUNC_( erfc )
DECLARE_FMUNARY_FUNC_( j0 )
DECLARE_FMUNARY_FUNC_( j1 )
DECLARE_FMUNARY_FUNC_( y0 )
DECLARE_FMUNARY_FUNC_( y1 )
DECLARE_FMUNARY_FUNC_( lgamma )
DECLARE_FMUNARY_FUNC_( rint )

DECLARE_FMBINARY_FUNC_( hypot )
#endif

// these are defined in misc/applicops.h
//
DECLARE_FMUNARY_FUNC_( pow2 )
DECLARE_FMUNARY_FUNC_( pow3 )
DECLARE_FMUNARY_FUNC_( pow4 )
DECLARE_FMUNARY_FUNC_( pow5 )
DECLARE_FMUNARY_FUNC_( pow6 )
DECLARE_FMUNARY_FUNC_( pow7 )

DECLARE_FMUNARY_FUNC_( pow8 )
#ifdef LTL_COMPLEX_MATH
// operations ONLY available on complex numbers
// argument complex<T>, but returning T
DECLARE_FMUNARY_FUNC_( abs)
DECLARE_FMUNARY_FUNC_( arg)
DECLARE_FMUNARY_FUNC_( norm )
DECLARE_FMUNARY_FUNC_( real )
DECLARE_FMUNARY_FUNC_( imag )
// argument complex<T> and returning complex<T>
DECLARE_FMUNARY_FUNC_( conj )
#endif

}

#endif // __LTL_FMEXPROPS__
