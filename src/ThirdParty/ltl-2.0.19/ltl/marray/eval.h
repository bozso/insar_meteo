/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: eval.h 541 2014-07-09 17:01:12Z drory $
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
#error "<ltl/marray/eval.h> must be included via <ltl/marray.h>, never alone!"
#endif

#ifndef __LTL_EVAL_EXPR__
#define __LTL_EVAL_EXPR__

#include <ltl/config.h>

#ifdef LTL_DEBUG_EXPRESSIONS
#include <iostream>
#endif

namespace ltl {

#ifdef LTL_USE_SIMD

//@{

/*!
 *   This helper class provides specializations for evaluating expressions that
 *   have vectrorized versions (through applicops_altivec.h).
 *
 *   If an expression is not vectorizable, i.e. it contains operations that have
 *   no vectorized version, we don't want any of the eval_vec methods to be called
 *   (since they are useless and probably might not even compile). Therefore, we
 *   use the compile time constant Expr::isVectorizable to pick a specialization
 *   of the eval_assign_expr_vec function, so that we do not instantiate any code
 *   that touches vector evaluations if the expression is not vectorizable.
 *
 */
template<bool Vectorizable>
struct eval_vectorizable
{
};

//! specialization for vectorizable expressions
template<>
struct eval_vectorizable<1>
{
      template<class T, int N, class E>
      static inline void eval_assign_expr_vec( MArray<T,N>& a, ExprNode<E,N>& e )
      {
         T* restrict_ dataptr = a.data();
         const int align = (int)((long)dataptr & 0x0FL);

         const int innerLoopNum = a.nelements();

         if( a.isStride1() && e.isStride1() && e.sameAlignmentAs( align ) )
         {
#ifdef LTL_DEBUG_EXPRESSIONS
            cerr << "Loop vectorized, alignment : " << align << endl;
#endif
            // good, both arrays have stride 1 and
            // they also have the same alignment in memory
            // so we can vectorize

            int j = 0;

            // handle the elements up to the next natural alignment boundary
            const int elemsPerVec = sizeof(typename VEC_TYPE(T))/sizeof(T);
            const int beforeAlign = (elemsPerVec - align/sizeof(T))%elemsPerVec;
            typename VEC_TYPE(T)* restrict_ vecptr_ = (typename VEC_TYPE(T)*)(dataptr+beforeAlign);

            for( ; j<beforeAlign; ++j )
               dataptr[j] = e.readWithoutStride(j);

            // now we can use the vector facilities
            e.alignData( beforeAlign ); // move the internal data pointer forward to alignment boundary
            const int vecLoopCount = (innerLoopNum - beforeAlign)/elemsPerVec;
            int k = 0;
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "   innerLoopNum :" << innerLoopNum << ", elemsPerVec:" << elemsPerVec << ", beforeAlign:" << beforeAlign << ", vecLoopCount:" << vecLoopCount << endl;
#endif

#ifdef LTL_UNROLL_EXPRESSIONS_SIMD
            // unroll the inner vector loop
            for( ; k<vecLoopCount-3; k+=4 )
            {
               // MArrayIter::readVec will prefetch (with GCC) 4 vectors ahead
               T_VEC_TYPE(T) tmp1 = (T_VEC_TYPE(T))e.readVec(k  );
               T_VEC_TYPE(T) tmp2 = (T_VEC_TYPE(T))e.readVec(k+1);
               T_VEC_TYPE(T) tmp3 = (T_VEC_TYPE(T))e.readVec(k+2);
               T_VEC_TYPE(T) tmp4 = (T_VEC_TYPE(T))e.readVec(k+3);
               vecptr_[k  ] = tmp1;
               vecptr_[k+1] = tmp2;
               vecptr_[k+2] = tmp3;
               vecptr_[k+3] = tmp4;
            }
#endif
            // handle the remaining elements after unrolling, or, if no unrolling,
            // execute the whole loop.
            for( ; k<vecLoopCount; ++k )
               vecptr_[k] = (typename VEC_TYPE(T))e.readVec(k);

            // handle the remainig elements (length mod vector length)
            e.alignData( -beforeAlign ); // move the internal data pointer back
            j += vecLoopCount*elemsPerVec;
            for( ; j<innerLoopNum; ++j )
               dataptr[j] = e.readWithoutStride(j);
         }
         else
            eval_assign_expr_1( a, e );
      }
};



/*!
 *  specialization for non-vectorizable expressions:
 *    just call standard eval_assign_expr_1()
 */
template<>
struct eval_vectorizable<0>
{
      template<class T, int N, class E>
      static inline void eval_assign_expr_vec( MArray<T,N>& a, ExprNode<E,N>& e )
      {
            return eval_assign_expr_1( a, e );
      }

};

//@}
#endif  //LTL_USE_SIMD


/*!
 *  This function is called from MArray::operator= ( ExprNode<>& e )
 *  to actually perform the evaluation and assignment
 */
template<class T, int N, class E>
inline void eval_assign_expr( MArray<T,N>& a, ExprNode<E,N>& e )
{
   CHECK_CONFORM( e, a );

   // first, we check for convolution operations
   // if present, we have to use a slow evaluation function without any loop
   // collapsing and optimized access patterns because we have to keep coherence
   // for accessing the expression elements at some offsets from the current
   // element for the convolution kernels to do their work ...
   if( ExprNode<E,N>::numConvolution != 0 )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "found " << ExprNode<E,N>::numConvolution << " convolution operators.\n";
#endif
      return eval_with_convolution( a, e );
   }

   // if there are index iterators (or partial reductions) present, we have no choice
   // but to use a slower evaluation function without any loop collapsing and optimized
   // access patterns, because we have to keep track of the actual indices for
   // the index iterators ...
   if( ExprNode<E,N>::numIndexIter != 0 )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "found " << ExprNode<E,N>::numIndexIter << " index iters.\n";
#endif
      return eval_with_index_iter( a, e );
   }

   // if the expression is 1-dimensional or if the memory layout of all
   // operands is contiguous, we can use fast 1-dimensional traversal
   // collapsing all loops
   if( N == 1 || (a.isStorageContiguous() && e.isStorageContiguous()) )
   {
#ifdef LTL_USE_SIMD
      // this will decide at compile time whether we have an expression consisting
      // only of terms which have vectorized implememtations. If so, we can
      // try to vectorize it. If not, just use the scalar code.
      return eval_vectorizable<ExprNode<E,N>::isVectorizable>::eval_assign_expr_vec( a, e );
#else
      // scalar version
      return eval_assign_expr_1( a, e );
#endif
   }
   else
   {
      // general case, we can't perform any of the (pseudo) 1-D optimizations
      // except in the innermost dimension (the one with the smallest stride).
      return eval_assign_expr_N( a, e );
   }
}



//! We are dealing with the 1 dimensional scalar case here
//  (see vectorized version above)
//
template<class T, int N, class E>
void eval_assign_expr_1( MArray<T,N>& a, ExprNode<E,N>& e )
{
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating with fully collapsed loops\n";
#endif

   const int innerLoopNum = a.nelements();
   T* restrict_ dataptr = a.data();

   if( a.isStride1() && e.isStride1() )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating with common stride 1\n";
#endif
      // good, both arrays have stride 1 and both arrays
      // also have contiguous memory

      int j=0;
#ifdef LTL_UNROLL_EXPRESSIONS
      for( ; j<innerLoopNum-3; j+=4 )
      {
         // gcc does poor aliasing analysis (in spite of use of restrict),
         // therefore we have to make it clear that writing to the result
         // array does not invalidate _any_ of the data associated with the
         // expression or its iterators (otherwise it reloads the data_
         // pointers of the iterators every time)    :-(
         typename E::value_type tmp1 = e.readWithoutStride(j  );
         typename E::value_type tmp2 = e.readWithoutStride(j+1);
         typename E::value_type tmp3 = e.readWithoutStride(j+2);
         typename E::value_type tmp4 = e.readWithoutStride(j+3);

         dataptr[j  ] = tmp1;
         dataptr[j+1] = tmp2;
         dataptr[j+2] = tmp3;
         dataptr[j+3] = tmp4;
      }
#endif
      // handle the remaining elements after unrolling, or, if no unrolling,
      // execute the whole loop.
      for( ; j<innerLoopNum; ++j )
         dataptr[j] = e.readWithoutStride(j);
   }
   else
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating without common stride\n";
#endif
      // well, then slightly less efficient
      const int stride = a.stride(1);
      int j=0, k=0;
#ifdef LTL_UNROLL_EXPRESSIONS
      for( ; j<innerLoopNum-3; j+=4, k+=4*stride )
      {
         typename E::value_type tmp1 = e.readWithStride(j  );
         typename E::value_type tmp2 = e.readWithStride(j+1);
         typename E::value_type tmp3 = e.readWithStride(j+2);
         typename E::value_type tmp4 = e.readWithStride(j+3);

         dataptr[k         ] = tmp1;
         dataptr[k+  stride] = tmp2;
         dataptr[k+2*stride] = tmp3;
         dataptr[k+3*stride] = tmp4;
      }
#endif
      // handle the remaining elements after unrolling, or, if no unrolling,
      // execute the whole loop.
      for( ; j<innerLoopNum; ++j, k+=stride )
         dataptr[j*stride] = e.readWithStride(j);
   }
}



//! this handles the N-dimensional case
//
template<class T, int N, class E>
void eval_assign_expr_N( MArray<T,N>& a, ExprNode<E,N>& e )
{
   // we already know that the storage ist not contiguous.
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating with stack traversal\n";
#endif

   const int innerLoopNum = a.length(1);
   int n = a.nelements();

   typename MArray<T,N>::iterator i = a.begin();

   if( a.isStride1() && e.isStride1() )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating with common stride 1\n";
#endif

      while( n )
      {
         T* restrict_ dataptr = i.data();

         int j=0;
         // optimize at least the innermost loop (smallest stride), optimizing away
         // advancing the iterators only at the end of the dimension and using
         // fast offset access during for the loop.
#ifdef LTL_UNROLL_EXPRESSIONS
         for( ; j<innerLoopNum-3; j+=4 )
         {
            // gcc does poor aliasing analysis (in spite of use of restrict),
            // therefore we have to make it clear that writing to the result
            // array does not invalidate _any_ of the data associated with the
            // expression or its iterators (otherwise it reloads the data_
            // pointers of the iterators every time)    :-(
            typename E::value_type tmp1 = e.readWithoutStride(j  );
            typename E::value_type tmp2 = e.readWithoutStride(j+1);
            typename E::value_type tmp3 = e.readWithoutStride(j+2);
            typename E::value_type tmp4 = e.readWithoutStride(j+3);

            dataptr[j  ] = tmp1;
            dataptr[j+1] = tmp2;
            dataptr[j+2] = tmp3;
            dataptr[j+3] = tmp4;
         }
#endif
         // handle the remaining elements after unrolling, or, if no unrolling,
         // execute the whole loop.
         for( ; j<innerLoopNum; ++j )
            dataptr[j] = e.readWithoutStride(j);

         i.advance( innerLoopNum );
         e.advance( innerLoopNum );
         i.advanceDim();
         e.advanceDim();
         n -= innerLoopNum;
      }
      return;
   }
   else
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating without common stride\n";
#endif

      const int stride = a.stride(1);

      while( n )
      {
         T* restrict_ dataptr = i.data();

         int j=0, k=0;
#ifdef LTL_UNROLL_EXPRESSIONS
         for( ; j<innerLoopNum-3; j+=4, k+=4*stride )
         {
            typename E::value_type tmp1 = e.readWithStride(j  );
            typename E::value_type tmp2 = e.readWithStride(j+1);
            typename E::value_type tmp3 = e.readWithStride(j+2);
            typename E::value_type tmp4 = e.readWithStride(j+3);

            dataptr[k         ] = tmp1;
            dataptr[k+  stride] = tmp2;
            dataptr[k+2*stride] = tmp3;
            dataptr[k+3*stride] = tmp4;
         }
#endif
         // handle the remaining elements after unrolling, or, if no unrolling,
         // execute the whole loop.
         for( ; j<innerLoopNum; ++j, k+=stride )
            dataptr[j*stride] = e.readWithStride(j);

         i.advance( innerLoopNum );
         e.advance( innerLoopNum );
         i.advanceDim();
         e.advanceDim();
         n -= innerLoopNum;
      }
   }
}


// if we have index expressions involved, we cannot use loop collapsing
// optimizations, since we need to keep track of where we are. we can only optimize
// the loop in the innermost dimension
//
template<class T, int N, class E>
void eval_with_index_iter( MArray<T,N>& a, ExprNode<E,N>& e )
{
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating with pure stack traversal due to IndexIter\n";
#endif
   // we have an index iterator in the expression,
   // so we cannot do much ...
   typename MArray<T,N>::iterator i = a.begin();
   const int innerLoopNum = a.length(1);
   if( i.isStride1() && e.isStride1() )
   {
      while( !i.done() )
      {
         for( int j=1; j<=innerLoopNum; ++j)
         {
            *i = *e;
            i.advanceWithStride1();
            e.advanceWithStride1();
         }
         if( N>1 ) // we're done in the N==1 case ...
         {
            //LTL_ASSERT_(i.needAdvanceDim(), "Iterator not at end of dim 1!");
            i.advanceDim();
            e.advanceDim();
         }
      }
   }
   else
   {
      while( !i.done() )
      {
         for( int j=1; j<=innerLoopNum; ++j)
         {
            *i = *e;
            i.advance();
            e.advance();
         }
         if( N>1 ) // we're done in the N==1 case ...
         {
            //LTL_ASSERT_(i.needAdvanceDim(), "Iterator not at end of dim 1!");
            i.advanceDim();
            e.advanceDim();
         }
      }
   }
}


template<typename T, int N, typename Expr>
void eval_with_convolution( MArray<T,N>& /*A*/, ExprNode<Expr,N>& /*E*/ )
{
   LTL_ASSERT_(false, "Reached unreachable code in eval_with_convolution");
}


template<typename T, typename Expr>
void eval_with_convolution( MArray<T,1>& A, ExprNode<Expr,1>& E )
{
   // we cannot do much optimization for the convolutions yet ...
   typename MArray<T,1>::iterator iter = A.begin();
   const int n = A.length(1);
   const int lb = ::abs(E.boundary_l(1));
   const int ub = ::abs(E.boundary_u(1));

#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "kernel boundaries : " << lb << ", " << ub << endl;
#endif

   int i;
   for( i=1; i<=lb; ++i ) { iter.advance(); E.advance(); } // advance past kernel size
   for( ; i<=n-ub; ++i )
   {
      *iter = *E;
      iter.advance();
      E.advance();
   }
}


template<typename T, typename Expr>
void eval_with_convolution( MArray<T,2>& A, ExprNode<Expr,2>& E )
{
   // we cannot do much optimization for the convolutions yet ...
   typename MArray<T,2>::iterator iter = A.begin();
   const int n1 = A.length(1);
   const int n2 = A.length(2);
   const int lb1 = ::abs(E.boundary_l(1));
   const int ub1 = ::abs(E.boundary_u(1));
   const int lb2 = ::abs(E.boundary_l(2));
   const int ub2 = ::abs(E.boundary_u(2));

#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "kernel boundaries : " << lb1 << ", " << ub1 << "  " << lb2 << ", " << ub2 << endl;
#endif

   int i, j;

   // first, advance past kernel size in dim 2
   for( j=1; j<=lb2; ++j )
   {
      iter.advance(n1);
      E.advance(n1);
      // end of dim 1, so advance
      //LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
      iter.advanceDim();
      E.advanceDim();
   }

   // now we are inside the allowed region in dim 2:
   for(; j<=n2-ub2; ++j )
   {
      // advance past kernel size in dim 1
      iter.advance(lb1);
      E.advance(lb1);

      // evaluate convolution expression
      for(i=lb1+1; i<=n1-ub1; ++i )
      {
         *iter = *E;
         iter.advance();
         E.advance();
      }

      // advance to end of dim 1
      iter.advance(ub1);
      E.advance(ub1);
      // end of dim 1, so advance
      //LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
      iter.advanceDim();
      E.advanceDim();
   }
}

template<typename T, typename Expr>
void eval_with_convolution( MArray<T,3>& A, ExprNode<Expr,3>& E )
{
   // we cannot do much optimization for the convolutions yet ...
   typename MArray<T,3>::iterator iter = A.begin();
   const int n1 = A.length(1);
   const int n2 = A.length(2);
   const int n3 = A.length(3);
   const int lb1 = ::abs(E.boundary_l(1));
   const int ub1 = ::abs(E.boundary_u(1));
   const int lb2 = ::abs(E.boundary_l(2));
   const int ub2 = ::abs(E.boundary_u(2));
   const int lb3 = ::abs(E.boundary_l(3));
   const int ub3 = ::abs(E.boundary_u(3));

#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "kernel boundaries : " << lb1 << ", " << ub1 << "  " << lb2 << ", " << ub2 << "  " << lb3 << ", " << ub3 << endl;
#endif

   int i, j, k;

   // first, advance past kernel size in dim 3
   for( k=1; k<=lb3; ++k )
      for( j=1; j<=n2; ++j )
      {
         iter.advance(n1);
         E.advance(n1);
         // end of dim 1, so advance
         LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
         iter.advanceDim();
         E.advanceDim();
      }

   // now we are inside the allowed region:
   for(k=lb3+1; k<=n3-ub3; ++k )
   {
      // advance past kernel size in dim 2
      for( j=1; j<=lb2; ++j )
      {
         iter.advance(n1);
         E.advance(n1);
         // end of dim 1, so advance
         //LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
         iter.advanceDim();
         E.advanceDim();
      }

      // now we are inside the allowed region in dim 2:
      for(; j<=n2-ub2; ++j )
      {
         // advance past kernel size in dim 1
         iter.advance(lb1);
         E.advance(lb1);

         // evaluate convolution expression
         for(i=lb1+1; i<=n1-ub1; ++i )
         {
            *iter = *E;
            iter.advance();
            E.advance();
         }

         // advance to end of dim 1
         iter.advance(ub1);
         E.advance(ub1);
         // end of dim 1, so advance
         //LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
         iter.advanceDim();
         E.advanceDim();
      }
      // advance to end of dim 2
      for( ; j<=n2; ++j )
      {
         iter.advance(n1);
         E.advance(n1);
         // end of dim 1, so advance
         //LTL_ASSERT_(iter.needAdvanceDim(), "Iterator not at end of dim 1!");
         iter.advanceDim();
         E.advanceDim();
      }
   }
}


}

#endif
