/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marray_io.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __LTL_MARRAYIO__
#define __LTL_MARRAYIO__

#include <ltl/config.h>

#include <string>
#include <iostream>

#include <ltl/marray.h>
#include <ltl/marray/expr_iter.h>
#include <ltl/misc/exceptions.h>
#include <ltl/misc/type_name.h>

using std::istream;
using std::ostream;
using std::string;
using std::endl;

namespace ltl {

template<class T, int N>
class MArray;
template<class T, int N>
class MArrayIter;
template<int N>
class Shape;
template<int N>
istream& operator>>( istream& is, Shape<N>& s );

/*! \addtogroup ma_stream_io
*/

//@{

/*! \relates ltl::MArray
  Write a 1-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,1>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,1> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,1> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",1> "<<*(A.shape()) << endl;
   os << "[ ";
   for (int i=1; i <= A.length(1); ++i, ++A)
   {
      os << *A << " ";
   }
   os << "]";

   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   return os;
}

/*! \relates ltl::MArray
  Write a 2-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,2>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,2> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,2> A = e2.begin();

   // rows x columns
   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",2> "<<*(A.shape()) << endl;
   os << "[";
   for (int j=1; j <= A.length(2); ++j)
   {
      os << "[ ";
      for (int i=1; i <= A.length(1); ++i, ++A)
      {
         os << *A << " ";
      }
      os << "]";
      if( j<A.length(2) )
         os << endl << " ";
   }

   os << "]";
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   return os;
}

/*! \relates ltl::MArray
  Write a 3-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,3>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,3> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,3> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",3> "<<*(A.shape()) << endl;
   os << "[";
   for( int k=1; k <= A.length(3); ++k)
   {
      os << "[";
      for (int j=1; j <= A.length(2); ++j)
      {
         os << "[ ";
         for (int i=1; i <= A.length(1); ++i, ++A)
         {
            os << *A << " ";
         }
         os << "]";
         if( j<A.length(2) )
            os << endl << "  ";
      }
      os << "]";
      if( k<A.length(3) )
         os << endl << " ";
   }

   os << "]";
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   return os;
}

/*! \relates ltl::MArray
  Write a 4-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,4>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,4> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,4> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",4> "<<*(A.shape()) << endl;
   os << "[";
   for( int l=1; l <= A.length(4); ++l)
   {
      os << "[";
      for( int k=1; k <= A.length(3); ++k)
      {
         os << "[";
         for (int j=1; j <= A.length(2); ++j)
         {
            os << "[ ";
            for (int i=1; i <= A.length(1); ++i, ++A)
            {
               os << *A << " ";
            }
            os << "]";
            if( j<A.length(2) )
               os << endl << "   ";
         }
         os << "]";
         if( k<A.length(3) )
            os << endl << "  ";
      }
      os << "]";
      if( l<A.length(4) )
         os << endl << " ";
   }

   os << "]";
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   return os;
}

/*! \relates ltl::MArray
  Write a 5-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,5>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,5> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,5> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",5> "<<*(A.shape()) << endl;
   os << "[";
   for( int m=1; m <= A.length(5); ++m)
   {
      os << "[";
      for( int l=1; l <= A.length(4); ++l)
      {
         os << "[";
         for( int k=1; k <= A.length(3); ++k)
         {
            os << "[";
            for (int j=1; j <= A.length(2); ++j)
            {
               os << "[ ";
               for (int i=1; i <= A.length(1); ++i, ++A)
               {
                  os << *A << " ";
               }
               os << "]";
               if( j<A.length(2) )
                  os << endl << "    ";
            }
            os << "]";
            if( k<A.length(3) )
               os << endl << "   ";
         }
         os << "]";
         if( l<A.length(4) )
            os << endl << "  ";
      }
      os << "]";
      if( m<A.length(5) )
         os << endl << " ";
   }

   os << "]";
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   return os;
}

/*! \relates ltl::MArray
  Write a 6-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,6>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,6> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,6> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",6> "<<*(A.shape()) << endl;
   os << "[";
   for( int n=1; n <= A.length(6); ++n)
   {
      os << "[";
      for( int m=1; m <= A.length(5); ++m)
      {
         os << "[";
         for( int l=1; l <= A.length(4); ++l)
         {
            os << "[";
            for( int k=1; k <= A.length(3); ++k)
            {
               os << "[";
               for (int j=1; j <= A.length(2); ++j)
               {
                  os << "[ ";
                  for (int i=1; i <= A.length(1); ++i, ++A)
                  {
                     os << *A << " ";
                  }
                  os << "]";
                  if( j<A.length(2) )
                     os << endl << "    ";
               }
               os << "]";
               if( k<A.length(3) )
                  os << endl << "   ";
            }
            os << "]";
            if( l<A.length(4) )
               os << endl << "  ";
         }
         os << "]";
         if( m<A.length(5) )
            os << endl << " ";
      }
      os << "]";
      if( n<A.length(6) )
         os << endl << " ";
   }
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   os << "]";

   return os;
}

/*! \relates ltl::MArray
  Write a 7-dim \c ltl::MArray to a stream.
*/
template<class Expr>
ostream& operator<<(ostream& os, const ExprBase<Expr,7>& E)
{
   typedef typename ExprNodeType<Expr>::expr_type ExprT;
   ExprNode<ExprT,7> e2(ExprNodeType<Expr>::node(E.derived()));
   ExprIter<ExprT,7> A = e2.begin();

   os << "MArray<"<<LTL_TYPE_NAME(typename ExprT::value_type)<<",7> "<<*(A.shape()) << endl;
   os << "[";
   for( int p=A.minIndex(7); p <= A.length(7); ++p)
   {
      os << "[";
      for( int n=A.minIndex(6); n <= A.length(6); ++n)
      {
         os << "[";
         for( int m=A.minIndex(5); m <= A.length(5); ++m)
         {
            os << "[";
            for( int l=A.minIndex(4); l <= A.length(4); ++l)
            {
               os << "[";
               for( int k=A.minIndex(3); k <= A.length(3); ++k)
               {
                  os << "[";
                  for (int j=A.minIndex(2); j <= A.length(2); ++j)
                  {
                     os << "[ ";
                     for (int i=A.minIndex(1); i <= A.length(1); ++i, ++A)
                     {
                        os << *A << " ";
                     }
                     os << "]";
                     if( j<A.length(2) )
                        os << endl << "    ";
                  }
                  os << "]";
                  if( k<A.length(3) )
                     os << endl << "   ";
               }
               os << "]";
               if( l<A.length(4) )
                  os << endl << "  ";
            }
            os << "]";
            if( m<A.length(5) )
               os << endl << " ";
         }
         os << "]";
         if( n<A.length(6) )
            os << endl << " ";
      }
      os << "]";
      if( p<A.length(7) )
         os << endl << " ";
   }
   LTL_ASSERT( A.done(), "ExprIter != end() at end of output");

   os << "]";

   return os;
}


/*! \relates ltl::MArray
  Read an \c ltl::MArray from a stream.
*/
template<class T, int N>
istream& operator>>( istream& is, MArray<T,N>& A )
{
   // read shape
   Shape<N> s;
   is >> s;
   A.realloc( s );

   T t;
   string tmp;

   typename MArray<T,N>::iterator i=A.begin();
   while( !i.done() )
   {
      is >> tmp;

      if( tmp[tmp.length()-1] != '[' )
         throw( IOException( "Format error while reading : '[' expected, got"+tmp ) );
      for( int n=A.minIndex(1); n<=A.maxIndex(1); ++n, ++i )
      {
         if( i.done() )
            throw( IOException( "File too long while reading MArray!" ) );
         is >> t;

         *i = t;
         if( is.bad() )
            throw( IOException( "Format error while reading MArray!" ) );
      }
      is >> tmp;

      if( tmp[tmp.length()-1] != ']' )
         throw( IOException( "Format error while reading : ']' expected, got"+tmp ) );
   }
   LTL_ASSERT( i.done(), "MArrayIter != end() at end of input");

   return is;
}

//@}

}

#endif // __LTL_MARRAYIO__

