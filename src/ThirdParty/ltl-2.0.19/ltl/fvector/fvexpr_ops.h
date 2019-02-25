/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvexpr_ops.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FVECTOR__
#error "<ltl/fvector/fexpr_ops.h> must be included via <ltl/fvector.h>, never alone!"
#endif


#ifndef __LTL_APPLICOPS_H__
#include <ltl/misc/applicops.h>
#endif

#ifndef __LTL_FVEXPR_OPS__
#define __LTL_FVEXPR_OPS__

namespace ltl {

//
// now actually define operator template functions
// using the macros defined in <ltl/fvector/fvexpr.h>
//

// standard operators
//
DECLARE_FVBINOP(+, TAdd)

DECLARE_FVBINOP(-, TSub)
DECLARE_FVBINOP(*, TMul)
DECLARE_FVBINOP(/, TDiv)
DECLARE_FVBINOP(&&, TAnd)
DECLARE_FVBINOP(||, TOr)

DECLARE_FVBINOP(& , TBitAnd )
DECLARE_FVBINOP(| , TBitOr )
DECLARE_FVBINOP(^ , TBitXor )
DECLARE_FVBINOP(% , TMod )

DECLARE_FVBINOP(> , TGT )
DECLARE_FVBINOP(< , TLT )
DECLARE_FVBINOP(>=, TGE )
DECLARE_FVBINOP(<=, TLE )
DECLARE_FVBINOP(!=, TNE )
DECLARE_FVBINOP(==, TEQ )

DECLARE_FVUNOP( +,  TPlus )
DECLARE_FVUNOP( -,  TMinus )
DECLARE_FVUNOP( !,  TNot )
DECLARE_FVUNOP( ~,  TNeg )


// standard library math functions
//
DECLARE_FVUNARY_FUNC_( sin )
DECLARE_FVUNARY_FUNC_( cos )
DECLARE_FVUNARY_FUNC_( tan )
DECLARE_FVUNARY_FUNC_( asin )
DECLARE_FVUNARY_FUNC_( acos )
DECLARE_FVUNARY_FUNC_( atan )
DECLARE_FVUNARY_FUNC_( sinh )
DECLARE_FVUNARY_FUNC_( cosh )
DECLARE_FVUNARY_FUNC_( tanh )
DECLARE_FVUNARY_FUNC_( exp )
DECLARE_FVUNARY_FUNC_( log )
DECLARE_FVUNARY_FUNC_( log10 )
DECLARE_FVUNARY_FUNC_( sqrt )
DECLARE_FVUNARY_FUNC_( fabs )
DECLARE_FVUNARY_FUNC_( floor )
DECLARE_FVUNARY_FUNC_( ceil )

DECLARE_FVBINARY_FUNC_( pow )
DECLARE_FVBINARY_FUNC_( fmod )
DECLARE_FVBINARY_FUNC_( atan2 )


// IEEE math functions
//
#ifdef HAVE_IEEE_MATH

DECLARE_FVUNARY_FUNC_( asinh )
DECLARE_FVUNARY_FUNC_( acosh )
DECLARE_FVUNARY_FUNC_( atanh )
DECLARE_FVUNARY_FUNC_( cbrt )
DECLARE_FVUNARY_FUNC_( expm1 )
DECLARE_FVUNARY_FUNC_( log1p )
DECLARE_FVUNARY_FUNC_( erf )
DECLARE_FVUNARY_FUNC_( erfc )
DECLARE_FVUNARY_FUNC_( j0 )
DECLARE_FVUNARY_FUNC_( j1 )
DECLARE_FVUNARY_FUNC_( y0 )
DECLARE_FVUNARY_FUNC_( y1 )
DECLARE_FVUNARY_FUNC_( lgamma )
DECLARE_FVUNARY_FUNC_( rint )

DECLARE_FVBINARY_FUNC_( hypot )
#endif

// these are defined in misc/applicops.h
//
DECLARE_FVUNARY_FUNC_( pow2 )
DECLARE_FVUNARY_FUNC_( pow3 )
DECLARE_FVUNARY_FUNC_( pow4 )
DECLARE_FVUNARY_FUNC_( pow5 )
DECLARE_FVUNARY_FUNC_( pow6 )
DECLARE_FVUNARY_FUNC_( pow7 )
DECLARE_FVUNARY_FUNC_( pow8 )

#ifdef LTL_COMPLEX_MATH
// operations ONLY available on complex numbers
// argument complex<T>, but returning T
DECLARE_FVUNARY_FUNC_( abs)
DECLARE_FVUNARY_FUNC_( arg)
DECLARE_FVUNARY_FUNC_( norm )
DECLARE_FVUNARY_FUNC_( real )
DECLARE_FVUNARY_FUNC_( imag )
// argument complex<T> and returning complex<T>
DECLARE_FVUNARY_FUNC_( conj )
#endif

}

#endif // __LTL_FVEXPROPS__
