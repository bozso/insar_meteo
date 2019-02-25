/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fvexpr.h 491 2011-09-02 19:36:39Z drory $
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
#error "<ltl/fvector/fvexpr.h> must be included via <ltl/fvector.h>, never alone!"
#endif


#ifndef __LTL_FVECOTR_EXPR__
#define __LTL_FVECTOR_EXPR__

#include <ltl/config.h>

namespace ltl {

//! Just to keep everything together in class browsers ...
class _et_fvector_parse_base
{ };

//
//! Binary operation node.
//
template<class A, class B, class Op, int N>
class FVExprBinopNode : public _et_fvector_parse_base
{
   private:
      A iter1_;
      B iter2_;

   public:
      typedef typename Op::value_type value_type;
      enum { static_size = A::static_size + B::static_size };

      inline FVExprBinopNode( const A& restrict_ a, const B& restrict_ b )
            : iter1_(a), iter2_(b)
      { }

      inline value_type operator[]( const int i ) const
      {
         return Op::eval( iter1_[i], iter2_[i] );
      }
};

//
//! Unary operation node.
//
template<class A, class Op, int N>
class FVExprUnopNode : public _et_fvector_parse_base
{
   private:
      A iter1_;

   public:
      typedef typename Op::value_type value_type;
      enum { static_size = A::static_size };

      inline FVExprUnopNode( const A& restrict_ a )
            : iter1_(a)
      { }

      inline value_type operator[]( const int i ) const
      {
         return Op::eval( iter1_[i] );
      }
};


//
//! Literal number.
//
template<class T>
class FVExprLiteralNode : public _et_fvector_parse_base
{
   private:
      const T f_;

   public:
      typedef T value_type;
      enum { static_size = 1 };

      inline FVExprLiteralNode( const T f ) : f_(f)
      { }

      inline const value_type operator[]( const int i ) const
      {
         return f_;
      }
};


//
//! The expression class itself.
//
template<class A, int N>
class FVExprNode : public _et_fvector_parse_base
{
   private:
      A iter_;

   public:
      typedef typename A::value_type value_type;
      enum { static_size = A::static_size };

      inline FVExprNode( const A& restrict_ a )
            : iter_(a)
      { }

      inline value_type operator[]( const int i ) const
      {
         return iter_[i];
      }
};


//
//! We need a trait class to deal with literals.
// Basically convert everything to an expression TExpr.
//
template<class T>
struct asFVExpr
{
   typedef FVExprLiteralNode<T> value_type;
};

//! Already an expression template term.
template<class T, int N>
struct asFVExpr< FVExprNode<T,N> >
{
   typedef FVExprNode<T,N> value_type;
};

//! An array operand.
template<class T, int N, int S>
struct asFVExpr< FVector<T,N,S> >
{
   typedef typename FVector<T,N,S>::const_iterator value_type;
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
//   vector  op  vector,
//   expr    op  vector,
//   vector  op  expr,
//   expr    op  expr,
//
//   vector  op  literal,
//   expr    op  literal,
//   literal op  vector,  and
//   literal op  expr
//
// a template version of the overloaded operator is provided
// for each case and for each operator

// vector , vector
#define FVBINOP_AA(operator, op)                                        \
template<class T1, class T2, int N, int S1, int S2>                     \
inline FVExprNode<FVExprBinopNode<typename FVector<T1,N,S1>::const_iterator, \
                            typename FVector<T2,N,S2>::const_iterator,  \
                            op<T1,T2>, N>, N >                          \
operator( const FVector<T1,N,S1>& a, const FVector<T2,N,S2>& b )        \
{                                                                       \
   typedef FVExprBinopNode<typename FVector<T1,N,S1>::const_iterator,   \
                        typename FVector<T2,N,S2>::const_iterator,      \
                        op<T1,T2>, N>                                   \
      ExprT;                                                            \
   return FVExprNode<ExprT,N>( ExprT(a.begin(), b.begin()) );           \
}

 
// expr , vector
#define FVBINOP_EA(operator,op)                                         \
template<class A, class T, int N, int S>                                \
inline FVExprNode<FVExprBinopNode<FVExprNode<A,N>,                      \
                            typename FVector<T,N,S>::const_iterator,    \
                            op <typename A::value_type,T>, N >, N >     \
operator( const FVExprNode<A,N>& a, const FVector<T,N,S>& b)            \
{                                                                       \
  typedef FVExprBinopNode<FVExprNode<A,N>,                              \
                       typename FVector<T,N,S>::const_iterator,         \
                       op <typename A::value_type,T>, N >               \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>( ExprT(a, b.begin()) );                    \
}

// vector , expr
#define FVBINOP_AE(operator, op)                                        \
template<class A, class T, int N, int S>                                \
inline FVExprNode<FVExprBinopNode<typename FVector<T,N,S>::const_iterator, \
                            FVExprNode<A,N>,                            \
                            op <T,typename A::value_type>, N >, N >     \
operator( const FVector<T,N,S>& a, const FVExprNode<A,N>& b )           \
{                                                                       \
  typedef FVExprBinopNode<typename FVector<T,N,S>::const_iterator,      \
                       FVExprNode<A,N>,                                 \
                       op <T,typename A::value_type>, N >               \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>( ExprT(a.begin(), b) );                    \
}

// expr , expr
#define FVBINOP_EE(operator, op)                                \
template<class A, class B, int N>                               \
inline FVExprNode<FVExprBinopNode<FVExprNode<A,N>,              \
                            FVExprNode<B,N>,                    \
                            op <typename A::value_type,         \
                                typename B::value_type>,        \
                            N >, N >                            \
operator( const FVExprNode<A,N>& a, const FVExprNode<B,N>& b )  \
{                                                               \
  typedef FVExprBinopNode<FVExprNode<A,N>,                      \
                       FVExprNode<B,N>,                         \
                       op <typename A::value_type,              \
                           typename B::value_type>,             \
                       N >                                      \
     ExprT;                                                     \
  return FVExprNode<ExprT,N>(ExprT(a,b));                       \
}

// array , literal
#define FVBINOP_AL(operator, op)                                        \
template<class T, int N, int S>                                         \
inline FVExprNode<FVExprBinopNode<typename FVector<T,N,S>::const_iterator, \
                                  FVExprLiteralNode<typename FVector<T,N,S>::value_type>, \
                                  op <T,T>, N >, N >                    \
operator( const FVector<T,N,S>& a, const typename FVector<T,N,S>::value_type& b ) \
{                                                                       \
  typedef FVExprBinopNode<typename FVector<T,N,S>::const_iterator,      \
                        FVExprLiteralNode<T>,                           \
                       op <T,T>, N >                                    \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>(ExprT(a.begin(),b));                       \
}

// expr , literal
#define FVBINOP_EL(operator, op)                                        \
template<class T, int N>                                                \
inline FVExprNode<FVExprBinopNode<FVExprNode<T,N>,                      \
                                  FVExprLiteralNode<typename T::value_type>, \
                                  op <typename T::value_type, typename T::value_type>, N >, N > \
operator( const FVExprNode<T,N>& a, const typename T::value_type& b )   \
{                                                                       \
  typedef FVExprBinopNode<FVExprNode<T,N>,                              \
      FVExprLiteralNode<typename T::value_type>,                        \
      op <typename T::value_type, typename T::value_type>, N >          \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>(ExprT(a,b));                               \
}

// literal , array
#define FVBINOP_LA(operator, op)                                        \
template<class T, int N, int S>                                         \
inline FVExprNode<FVExprBinopNode<FVExprLiteralNode<typename FVector<T,N,S>::value_type>, \
                                  typename FVector<T,N,S>::const_iterator, \
                                  op <T, T>, N >, N >                   \
operator( const typename FVector<T,N,S>::value_type& a, const FVector<T,N,S>& b ) \
{                                                                       \
  typedef FVExprBinopNode<FVExprLiteralNode<T>,                         \
     typename FVector<T,N,S>::const_iterator,                           \
     op <T, T>, N >                                                     \
    ExprT;                                                              \
  return FVExprNode<ExprT,N>( ExprT(a, b.begin()) );                    \
}

// literal , expr
#define FVBINOP_LE(operator, op)                                        \
template<class T, int N>                                                \
inline FVExprNode<FVExprBinopNode<FVExprLiteralNode<typename T::value_type>, \
                                  FVExprNode<T,N>,                      \
                                  op <typename T::value_type, typename T::value_type>, N >, N > \
operator( const typename T::value_type& a, const FVExprNode<T,N>& b )   \
{                                                                       \
  typedef FVExprBinopNode<FVExprLiteralNode<typename T::value_type>,    \
      FVExprNode<T,N>,                                                  \
      op <typename T::value_type, typename T::value_type>, N >          \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>(ExprT(a,b));                               \
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
#define FVUNOP_A(operator, op)                                          \
template<class T,int N, int S>                                          \
inline FVExprNode<FVExprUnopNode<typename FVector<T,N,S>::const_iterator, \
                           op <T>, N >, N >                             \
operator( const FVector<T,N,S>& a )                                     \
{                                                                       \
  typedef FVExprUnopNode<typename FVector<T,N,S>::const_iterator, op <T>,N> \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>( ExprT(a.begin()) );                       \
}


// expr
#define FVUNOP_E(operator, op)                                          \
template<class A, int N>                                                \
inline FVExprNode<FVExprUnopNode<FVExprNode<A,N>,                       \
                           op <typename A::value_type>, N >, N >        \
operator( const FVExprNode<A,N>& a )                                    \
{                                                                       \
  typedef FVExprUnopNode<FVExprNode<A,N>, op <typename A::value_type>, N > \
     ExprT;                                                             \
  return FVExprNode<ExprT,N>(ExprT(a));                                 \
}




// convenience macro to define the parse tree templates
// (the applicative templates for all standard unary and
// binary operations are defined in misc/applicops.h, since
// those are common to MArray, FVector, and FMatrix

// Arguments: binary operator, arbitrary name
//
#define DECLARE_FVBINOP(operation, opname)                \
   FVBINOP_AA(operator  operation,__ltl_##opname)         \
   FVBINOP_AE(operator  operation,__ltl_##opname)         \
   FVBINOP_EA(operator  operation,__ltl_##opname)         \
   FVBINOP_EE(operator  operation,__ltl_##opname)         \
   FVBINOP_AL(operator  operation,__ltl_##opname)         \
   FVBINOP_EL(operator  operation,__ltl_##opname)         \
   FVBINOP_LA(operator  operation,__ltl_##opname)         \
   FVBINOP_LE(operator  operation,__ltl_##opname)



// Arguments: unary operator, arbitrary name
//
#define DECLARE_FVUNOP(operation, opname)         \
   FVUNOP_A(operator  operation,__ltl_##opname)   \
   FVUNOP_E(operator  operation,__ltl_##opname)



// Arguments: function, return type
//
#define DECLARE_FVBINARY_FUNC_(function)                \
   FVBINOP_AA(function,__ltl_##function)                \
   FVBINOP_AE(function,__ltl_##function)                \
   FVBINOP_EA(function,__ltl_##function)                \
   FVBINOP_EE(function,__ltl_##function)                \
   FVBINOP_AL(function,__ltl_##function)                \
   FVBINOP_EL(function,__ltl_##function)                \
   FVBINOP_LA(function,__ltl_##function)                \
   FVBINOP_LE(function,__ltl_##function)

// Arguments: function, return type
//
#define DECLARE_FVUNARY_FUNC_(function)                        \
FVUNOP_A(function,__ltl_##function)                            \
FVUNOP_E(function,__ltl_##function)


// Finally, to make is easier for the user to declare their
// own functions for use in expression templates, provide
// single macros that declare both the applicative templates
// and the parse tree nodes
//
#define DECLARE_FV_BINARY_FUNC(function, ret_type)            \
MAKE_BINAP_FUNC( __ltl_##function, ret_type, function )       \
DECLARE_FVBINARY_FUNC_(function)

#define DECLARE_FV_UNARY_FUNC(function, ret_type)             \
MAKE_UNAP_FUNC( __ltl_##function, ret_type, function )        \
DECLARE_FVUNARY_FUNC_(function)

}

#endif // __LTL_FVECTOR_EXPR__

