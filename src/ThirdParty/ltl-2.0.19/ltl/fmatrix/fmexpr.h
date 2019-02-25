/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fmexpr.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/fmatrix/fmexpr.h> must be included via <ltl/fmatrix.h>, never alone!"
#endif


#ifndef __LTL_FMATRIX_EXPR__
#define __LTL_FMATRIX_EXPR__

#include <ltl/config.h>

namespace ltl {

template<class T, int M, int N> class FMatrix;


//! Just to keep everything together in class browsers...
class _et_fmatrix_parse_base
{ };

//
//! Binary operation node.
//
template<class A, class B, class Op, int M, int N>
class FMExprBinopNode : public _et_fmatrix_parse_base
{
   private:
      A iter1_;
      B iter2_;

   public:
      typedef typename Op::value_type value_type;
      enum { static_size = A::static_size + B::static_size };

      inline FMExprBinopNode( const A& a, const B& b )
            : iter1_(a), iter2_(b)
      { }

      inline value_type operator()( const int i, const int j ) const
      {
         return Op::eval( iter1_(i,j), iter2_(i,j) );
      }

      inline value_type operator[]( const int i ) const
      {
         return Op::eval( iter1_[i], iter2_[i] );
      }
};

//
//! Unary operation node.
//
template<class A, class Op, int M, int N>
class FMExprUnopNode : public _et_fmatrix_parse_base
{
   private:
      A iter1_;

   public:
      typedef typename Op::value_type value_type;
      enum { static_size = A::static_size };

      inline FMExprUnopNode( const A& a )
            : iter1_(a)
      { }

      inline value_type operator()( const int i, const int j ) const
      {
         return Op::eval( iter1_(i,j) );
      }

      inline value_type operator[]( const int i ) const
      {
         return Op::eval( iter1_[i] );
      }
};


//
//! Literal number.
//
template<class T>
class FMExprLiteralNode : public _et_fmatrix_parse_base
{
   private:
      const T f_;

   public:
      typedef T value_type;
      enum { static_size = 1 };

      inline FMExprLiteralNode( const T f ) : f_(f)
      { }

      inline const value_type operator()( const int i, const int j ) const
      {
         return f_;
      }

      inline const value_type operator[]( const int i ) const
      {
         return f_;
      }
};


//
//! Now the expression class itself.
//
template<class A, int M, int N>
class FMExprNode : public _et_fmatrix_parse_base
{
   private:
      A iter_;

   public:
      typedef typename A::value_type value_type;
      enum { static_size = A::static_size };

      inline FMExprNode( const A& a )
            : iter_(a)
      { }

      inline value_type operator()( const int i, const int j ) const
      {
         return iter_(i,j);
      }

      inline value_type operator[]( const int i ) const
      {
         return iter_[i];
      }
};


//
//! We need a trait class to deal with literals.
/*!
  Basically convert everything to an expression TExpr.
*/
template<class T>
struct asFMExpr
{
   typedef FMExprLiteralNode<T> value_type;
};

//! Already an expression template term.
template<class T, int M, int N>
struct asFMExpr< FMExprNode<T,M,N> >
{
   typedef FMExprNode<T,M,N> value_type;
};

//! An array operand.
template<class T, int M, int N>
struct asFMExpr< FMatrix<T,M,N> >
{
   typedef typename FMatrix<T,M,N>::const_iterator value_type;
};


//
// that's all ...
// the rest are the operator templates
//


//
// BINARY OPERATOR PARSE TREE TEMPLATES
//
// there are basically 8 cases for binary ops, namely:
//
//   matrix  op  matrix,
//   expr    op  matrix,
//   matrix  op  expr,
//   expr    op  expr,
//
//   matrix  op  literal,
//   expr    op  literal,
//   literal op  matrix,  and
//   literal op  expr
//
// a template version of the overloaded operator is provided
// for each case and for each operator

// matrix , matrix
#define FMBINOP_AA(operator, op)                                        \
template<class T1, class T2, int M, int N>                              \
inline FMExprNode<FMExprBinopNode<typename FMatrix<T1,M,N>::const_iterator, \
                                  typename FMatrix<T2,M,N>::const_iterator, \
                                  op<T1,T2>, M, N>, M, N >              \
operator( const FMatrix<T1,M,N>& a, const FMatrix<T2,M,N>& b )          \
{                                                                       \
   typedef FMExprBinopNode<typename FMatrix<T1,M,N>::const_iterator,    \
      typename FMatrix<T2,M,N>::const_iterator,                         \
      op<T1,T2>, M, N >                                                 \
     ExprT;                                                             \
   return FMExprNode<ExprT,M,N>( ExprT(a.begin(), b.begin()) );         \
}

 
// expr , matrix
#define FMBINOP_EA(operator,op)                                         \
template<class A, class T, int M, int N>                                \
inline FMExprNode<FMExprBinopNode<FMExprNode<A,M,N>,                    \
                                  typename FMatrix<T,M,N>::const_iterator, \
                                  op <typename A::value_type,T>, M,N >, M,N > \
operator( const FMExprNode<A,M,N>& a, const FMatrix<T,M,N>& b)          \
{                                                                       \
  typedef FMExprBinopNode<FMExprNode<A,M,N>,                            \
      typename FMatrix<T,M,N>::const_iterator,                          \
      op <typename A::value_type,T>, M, N >                             \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>( ExprT(a, b.begin()) );                  \
}

// array , expr
#define FMBINOP_AE(operator, op)                                        \
template<class A, class T, int M, int N>                                \
inline FMExprNode<FMExprBinopNode<typename FMatrix<T,M,N>::const_iterator, \
                                  FMExprNode<A,M,N>,                    \
                                  op <T,typename A::value_type>,M,N >,M,N > \
operator( const FMatrix<T,M,N>& a, const FMExprNode<A,M,N>& b )         \
{                                                                       \
  typedef FMExprBinopNode<typename FMatrix<T,M,N>::const_iterator,      \
      FMExprNode<A,M,N>,                                                \
      op <T,typename A::value_type>,M,N >                               \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>( ExprT( a.begin(), b) );                 \
}

// expr , expr
#define FMBINOP_EE(operator, op)                                        \
template<class A, class B, int M, int N>                                \
inline FMExprNode<FMExprBinopNode<FMExprNode<A,M,N>,                    \
                                  FMExprNode<B,M,N>,                    \
                                  op <typename A::value_type, typename B::value_type>, \
                                  M,N >, M,N >                          \
operator( const FMExprNode<A,M,N>& a, const FMExprNode<B,M,N>& b )      \
{                                                                       \
  typedef FMExprBinopNode<FMExprNode<A,M,N>,                            \
      FMExprNode<B,M,N>,                                                \
      op <typename A::value_type, typename B::value_type>, M,N >        \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>(ExprT(a,b));                             \
}

// array , literal
#define FMBINOP_AL(operator, op)                                        \
template<class T, int M, int N>                                         \
inline FMExprNode<FMExprBinopNode<typename FMatrix<T,M,N>::const_iterator, \
                                  FMExprLiteralNode<typename FMatrix<T,M,N>::value_type>, \
                                  op <T,T>, M,N >, M,N >                \
operator( const FMatrix<T,M,N>& a, const typename FMatrix<T,M,N>::value_type& b ) \
{                                                                       \
  typedef FMExprBinopNode<typename FMatrix<T,M,N>::const_iterator,      \
      FMExprLiteralNode<T>,                                             \
      op <T,T>, M,N >                                                   \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>( ExprT(a.begin(),b) );                   \
}

// expr , literal
#define FMBINOP_EL(operator, op)                                        \
template<class T, int M, int N>                                         \
inline FMExprNode<FMExprBinopNode<FMExprNode<T,M,N>,                    \
                                  FMExprLiteralNode<typename T::value_type>, \
                                  op <typename T::value_type, typename T::value_type>,M,N >,M,N > \
operator( const FMExprNode<T,M,N>& a, const typename T::value_type& b ) \
{                                                                       \
  typedef FMExprBinopNode<FMExprNode<T,M,N>,                            \
      FMExprLiteralNode<typename T::value_type>,                        \
      op <typename T::value_type, typename T::value_type>, M,N >        \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>(ExprT(a,b));                             \
}

// literal , array
#define FMBINOP_LA(operator, op)                                        \
template<class T, int M, int N>                                         \
inline FMExprNode<FMExprBinopNode<FMExprLiteralNode<typename FMatrix<T,M,N>::value_type>, \
                                  typename FMatrix<T,M,N>::const_iterator, \
                                  op <T, T>, M,N >, M,N >               \
operator( const typename FMatrix<T,M,N>::value_type& a, const FMatrix<T,M,N>& b ) \
{                                                                       \
  typedef FMExprBinopNode<FMExprLiteralNode<T>,                         \
      typename FMatrix<T,M,N>::const_iterator,                          \
      op <T, T>, M,N >                                                  \
    ExprT;                                                              \
  return FMExprNode<ExprT,M,N>(ExprT(a, b.begin()));                    \
}

// literal , expr
#define FMBINOP_LE(operator, op)                                        \
template<class T, int M, int N>                                         \
inline FMExprNode<FMExprBinopNode<FMExprLiteralNode<typename T::value_type>, \
                                  FMExprNode<T,M,N>,                    \
                                  op <typename T::value_type, typename T::value_type>,M,N >,M,N > \
operator( const typename T::value_type& a, const FMExprNode<T,M,N>& b ) \
{                                                                       \
  typedef FMExprBinopNode<FMExprLiteralNode<typename T::value_type>,    \
      FMExprNode<T,M,N>,                                                \
      op <typename T::value_type, typename T::value_type>,M,N >         \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>(ExprT(a,b));                             \
}

//
// UNARY OPERATOR PARSE TREE TEMPLATES
//
// there are 2 cases for unary ops, namely:
//       op array,
// and   op expr.
// a template function operator is provided
// for each case and for each operator

// array
#define FMUNOP_A(operator, op)                                          \
template<class T, int M, int N>                                         \
inline FMExprNode<FMExprUnopNode<typename FMatrix<T,M,N>::const_iterator, \
                                 op <T>, M,N >, M,N >                   \
operator( const FMatrix<T,M,N>& a )                                     \
{                                                                       \
  typedef FMExprUnopNode<typename FMatrix<T,M,N>::const_iterator,       \
     op <T>, M,N >                                                      \
   ExprT;                                                               \
  return FMExprNode<ExprT,M,N>(ExprT(a.begin()));                       \
}


// expr
#define FMUNOP_E(operator, op)                                          \
template<class A, int M, int N>                                         \
inline FMExprNode<FMExprUnopNode<FMExprNode<A,M,N>,                     \
                                 op <typename A::value_type>, M,N >, M,N > \
operator( const FMExprNode<A,M,N>& a )                                  \
{                                                                       \
  typedef FMExprUnopNode<FMExprNode<A,M,N>, op <typename A::value_type>,M,N > \
     ExprT;                                                             \
  return FMExprNode<ExprT,M,N>(ExprT(a));                               \
}




// convenience macro to define the parse tree templates
// (the applicative templates for all standard unary and
// binary operations are defined in misc/applicops.h, since
// those are common to MArray, FVector, and FMatrix

// Arguments: binary operator, arbitrary name
//
#define DECLARE_FMBINOP(operation, opname)                \
   FMBINOP_AA(operator  operation,__ltl_##opname)         \
   FMBINOP_AE(operator  operation,__ltl_##opname)         \
   FMBINOP_EA(operator  operation,__ltl_##opname)         \
   FMBINOP_EE(operator  operation,__ltl_##opname)         \
   FMBINOP_AL(operator  operation,__ltl_##opname)         \
   FMBINOP_EL(operator  operation,__ltl_##opname)         \
   FMBINOP_LA(operator  operation,__ltl_##opname)         \
   FMBINOP_LE(operator  operation,__ltl_##opname)



// Arguments: unary operator, arbitrary name
//
#define DECLARE_FMUNOP(operation, opname)         \
   FMUNOP_A(operator  operation,__ltl_##opname)   \
   FMUNOP_E(operator  operation,__ltl_##opname)



// Arguments: function, return type
//
#define DECLARE_FMBINARY_FUNC_(function)                \
   FMBINOP_AA(function,__ltl_##function)                \
   FMBINOP_AE(function,__ltl_##function)                \
   FMBINOP_EA(function,__ltl_##function)                \
   FMBINOP_EE(function,__ltl_##function)                \
   FMBINOP_AL(function,__ltl_##function)                \
   FMBINOP_EL(function,__ltl_##function)                \
   FMBINOP_LA(function,__ltl_##function)                \
   FMBINOP_LE(function,__ltl_##function)



// Arguments: function, return type
//
#define DECLARE_FMUNARY_FUNC_(function)                 \
FMUNOP_A(function,__ltl_##function)                     \
FMUNOP_E(function,__ltl_##function)


// Finally, to make is easier for the user to declare their
// own functions for use in expression templates, provide
// single macros that declare both the applicative templates
// and the parse tree nodes
//
#define DECLARE_FM_BINARY_FUNC(function, ret_type)            \
MAKE_BINAP_FUNC( __ltl_##function, ret_type, function )       \
DECLARE_FMBINARY_FUNC_(function)

#define DECLARE_FM_UNARY_FUNC(function, ret_type)             \
MAKE_UNAP_FUNC( __ltl_##function, ret_type, function )        \
DECLARE_FMUNARY_FUNC_(function)

}

#endif // __LTL_FMATRIX_EXPR__

