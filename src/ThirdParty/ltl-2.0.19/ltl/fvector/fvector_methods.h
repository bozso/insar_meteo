/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvector_methods.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
 *                         Claus A. Goessl <cag@usm.uni-muenchen.de>
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
#error "<ltl/fvector/fvector_methods.h> must be included via <ltl/fvector.h>, never alone!"
#endif



#ifndef __LTL_FVECTOR_METHODS__
#define __LTL_FVECTOR_METHODS__

#include <ltl/config.h>

namespace ltl {

template<class T, int N, int S>
inline FVector<T,N,S>::FVector( const FVector<T,N,S>& other )
   : FVMemory<T,N,S>( other )
{ (*this) = other; }


/*! In the general case, we become a reference to T* a
  for S=0 (the default parameter) we copy the contents of a
  these two cases are distinguished in FVMemory constructor
*/
template<class T, int N, int S>
inline FVector<T,N,S>::FVector( T* const a )
   : FVMemory<T,N,S>( a )
{ }

template<class T, int N, int S>
inline FVector<T,N,S>::FVector( const T t )
{
   fill(t);
//    tFVLoop< FVector<T,N,S>, FVExprLiteralNode<T>,
//             fv_equ_assign<T,T>, 
//             N >::eval( *this, FVExprLiteralNode<T>(t) );   
}


//! Construct from expression.
template<class T, int N, int S>
template<class Expr>
inline FVector<T,N,S>::FVector( const FVExprNode<Expr,N>& e )
{
   (*this) = e;   
//    tFVLoop< FVector<T,N,S>, FVExprNode<Expr,N>,
//             fv_equ_assign<T,typename FVExprNode<Expr,N>::value_type>, 
//             N >::eval( *this, e );
}


template<class T, int N, int S>
inline void FVector<T,N,S>::fill( const T x )
{
// copy from constructor
   tFVLoop< FVector<T,N,S>, FVExprLiteralNode<T>,
            fv_equ_assign<T,T>, 
            N >::eval( *this, FVExprLiteralNode<T>(x) );

// orig Niv:
//    for( int i=0; i<N; ++i )
//       (*this)[i] = x;   
}


template<class T, int N, int S>
template<class T2, int S2>
inline void FVector<T,N,S>::swap( FVector<T2, N, S2>& other )
{
   tFVSwap< FVector<T, N, S>, FVector<T2, N, S2>, N >::eval( *this, other );   
}

}

#endif // __LTL_FVECTOR_METHODS__
