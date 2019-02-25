/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmatrix_methods.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __LTL_IN_FILE_FMATRIX__
#error "<ltl/fmatrix/fmatrix_methods.h> must be included via <ltl/fmatrix.h>, never alone!"
#endif



#ifndef __LTL_FMATRIX_METHODS__
#define __LTL_FMATRIX_METHODS__

#include <ltl/config.h>

namespace ltl
{

template<class T, int M, int N>
inline FMatrix<T,M,N>::FMatrix( const FMatrix<T,M,N>& restrict_ other )
{
   (*this) = other;   
//    for( int i=0; i<M*N; ++i )
//       __data_[i] = other.__data_[i];
}

template<class T, int M, int N>
inline FMatrix<T,M,N>::FMatrix( const T* restrict_ t )
{
   for( int i=0; i<M*N; ++i )
      __data_[i] = *(t++);
}

template<class T, int M, int N>
inline FMatrix<T,M,N>::FMatrix( const T t )
{
   fill(t);
}
      
template<class T, int M, int N>
template<class Expr>
inline FMatrix<T,M,N>::FMatrix( const FMExprNode<Expr,M,N>& e )
{
   (*this) = e;
}

template<class T, int M, int N>
inline void FMatrix<T,M,N>::fill( const T x )
{
   tFMLoop< FMatrix<T,M,N>, FMExprLiteralNode<T>,
            fm_equ_assign<T,T>, 
            M, N >::eval( *this, FMExprLiteralNode<T>(x) );

//    for( int i=0; i<M*N; ++i )
//       __data_[i] = x;
}
      
template<class T, int M, int N>
inline void FMatrix<T,M,N>::swapRows(const int row1, const int row2)
{
   LTL_ASSERT( row1>0 && row1<=M && row2>0 && row2<=M, 
               "Row index out of bounds in FMatrix.swapRows(). Index : "
               <<row1<<" or "<<row2<<" Range 1-"<<M );
//    if(row1 != row2)
//    {
//       RowVector r1 = row(row1);
//       FVector<T,N> dummy;
//       dummy = r1;
//       RowVector r2 = row(row2);
//       r1 = r2;
//       r2 = dummy;
      RowVector r1 = row(row1);
      RowVector r2 = row(row2);
      r1.swap(r2);
//    }
}

template<class T, int M, int N>
inline void FMatrix<T,M,N>::swapCols(const int col1, const int col2)
{
   LTL_ASSERT( col1>0 && col1<=N && col2>0 && col2<=N, 
               "Col index out of bounds in FMatrix.swapCols(). Index : "
               <<col1<<" or "<<col2<<" Range 1-"<<N );
//    if(col1 != col2)
//    {
//       ColumnVector c1 = col(col1);
//       FVector<T,N> dummy;
//       dummy = c1;
//       ColumnVector c2 = col(col2);
//       c1 = c2;
//       c2 = dummy;
      ColumnVector c1 = col(col1);
      ColumnVector c2 = col(col2);
      c1.swap(c2);
//    }
}

}

#endif // __LTL_FMATRIX_METHODS__
