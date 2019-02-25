/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: eval_reduc.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __LTL_REDUCTIONS__
#error "<ltl/marray/eval_reduc.h> must be included via <ltl/statistics.h>, never alone!"
#endif


#ifndef __LTL_REDUCE__
#define __LTL_REDUCE__

#include <ltl/marray/shape_iter.h>

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
struct eval_reduc_vectorizable
{
};

//! specialization for vectorizable expressions
template<>
struct eval_reduc_vectorizable<1>
{
   template<int N, class E, class Reduction>
   static void eval_full_reduction_vec( ExprNode<E,N> e, Reduction& R )
   {
      // copy the Reduction object to our local stack frame
      // so that the compiler sees that it can hold the members in registers
      // avoiding writing back to memory after every loop step
      typename Reduction::vec_reduction R1(R);

      const int align = e.getAlignment(); // alignment relative to vector length

      if( e.isStride1() && e.sameAlignmentAs(align) )
      {
   #ifdef LTL_DEBUG_EXPRESSIONS
         cerr << "evaluating full reduction vectorized\n";
   #endif
         const int innerLoopNum = e.shape()->nelements();

         // handle the elements up to the next natural alignment boundary
         const int elemsPerVec = sizeof(typename VEC_TYPE(typename Reduction::vec_calc_type))/sizeof(typename Reduction::vec_calc_type);

         const int beforeAlign = (elemsPerVec - align/sizeof(typename Reduction::vec_calc_type))%elemsPerVec;

         int j=0;
         int k = 0;
         const int vecLoopCount = (innerLoopNum - beforeAlign)/elemsPerVec;

#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "   innerLoopNum :" << innerLoopNum << ", elemsPerVec:" << elemsPerVec
           << ", beforeAlign:" << beforeAlign << ", vecLoopCount:" << vecLoopCount << endl;
#endif
         for( ; j<beforeAlign; ++j )
            if( !R1.evaluate( e.readWithoutStride(j) ) )
                goto loop_end;

         e.alignData( beforeAlign );
#ifdef LTL_UNROLL_EXPRESSIONS_SIMD
         // unroll the inner vector loop
         for( ; k<vecLoopCount-3; k+=4 )
         {
            if( !R1.evaluate( e.readVec(k) ) )
               goto loop_end;
            if( !R1.evaluate( e.readVec(k+1) ) )
               goto loop_end;
            if( !R1.evaluate( e.readVec(k+2) ) )
               goto loop_end;
            if( !R1.evaluate( e.readVec(k+3) ) )
               goto loop_end;
         }
#endif
         // handle the remaining elements after unrolling, or, if no unrolling,
         // execute the whole loop.
         for( ; k<vecLoopCount; ++k )
            if( !R1.evaluate( e.readVec(k) ) )
               goto loop_end;

         // handle the remainig elements (length mod vector length)
         e.alignData( -beforeAlign );
         j += vecLoopCount*elemsPerVec;
         for( ; j<innerLoopNum; ++j )
            if( !R1.evaluate( e.readWithoutStride(j) ) )
               goto loop_end;

         loop_end:
         R.copyResult( R1 );
      }
      else // stride != 1
      {
         return eval_full_reduction_1( e, R );
      }
   }
};



/*!
 *  specialization for non-vectorizable expressions:
 *    just call standard eval_assign_expr_1()
 */
template<>
struct eval_reduc_vectorizable<0>
{
   template<int N, class E, class Reduction>
   static inline void eval_full_reduction_vec( ExprNode<E,N> e, Reduction& R )
   {
      return eval_full_reduction_1( e, R );
   }
};

//@}
#endif  //LTL_USE_SIMD


template<int N, class E, class Reduction>
inline void eval_full_reduction( ExprNode<E,N> e, Reduction& R )
{
   if( ExprNode<E,N>::numIndexIter != 0 )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "found " << ExprNode<E,N>::numIndexIter << " index iters\n";
#endif

      return eval_full_reduction_with_index( e, R );
   }

   // if the expression is 1-dimensional or if the memory layout of all
   // operands is contiguous, we can use fast 1-dimensional traversal
   if( N == 1 || e.isStorageContiguous() )
   {
#ifdef LTL_USE_SIMD
      // this will decide at compile time whether we have an expression consisting
      // only of terms which have vectorized implememtations. If so, we can
      // try to vectorize it. If not, just use the scalar code.
      return eval_reduc_vectorizable<ExprNode<E,N>::isVectorizable
                                     && Reduction::isVectorizable>::eval_full_reduction_vec( e, R );
#else
      // scalar version
      return eval_full_reduction_1( e, R );
#endif
   }
   else
      return eval_full_reduction_N( e, R );
}


// we are dealing with the 1 dimensional case
//
template<int N, class E, class Reduction>
void eval_full_reduction_1( ExprNode<E,N> e, Reduction& R )
{
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating full reduction with collapsed loops\n";
#endif

   // copy the Reduction object to our local stack frame
   // so that the compiler sees that it can hold the members in registers
   // avoiding writing back to memory after every loop step
   Reduction R1(R);

   int innerLoopNum = e.shape()->nelements();

   if( e.isStride1() )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating full reduction with stride1\n";
#endif
      // good, both arrays have stride 1, this means both arrays
      // also have contiguous memory

      int j=0;
#ifdef LTL_UNROLL_EXPRESSIONS
      for( ; j<innerLoopNum-3; j+=4 )
      {
         if( !R1.evaluate( e.readWithoutStride(j) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithoutStride(j+1) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithoutStride(j+2) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithoutStride(j+3) ) )
            goto loop_end;
      }
#endif
      for( ; j<innerLoopNum; ++j )
          if( !R1.evaluate( e.readWithoutStride(j) ) )
             goto loop_end;
   }
   else // stride != 1
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating full reduction without common stride\n";
#endif
      // well, then slightly less efficient
      int j=0;
#ifdef LTL_UNROLL_EXPRESSIONS
      for( ; j<innerLoopNum-3; j+=4 )
      {
         if( !R1.evaluate( e.readWithStride(j) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithStride(j+1) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithStride(j+2) ) )
            goto loop_end;
         if( !R1.evaluate( e.readWithStride(j+3) ) )
            goto loop_end;
      }
#endif
      for( ; j<innerLoopNum; ++j )
         if( !R1.evaluate( e.readWithStride(j) ) )
            goto loop_end;
   }
   loop_end:
   // copy results back
   R.copyResult( R1 );
}



// now the N-dimensional case
//
template<int N, class E, class Reduction>
void eval_full_reduction_N( ExprNode<E,N> e, Reduction& R )
{
   // we already know that the storage ist not contiguous.
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating full reduction with stack traversal\n";
#endif

   // copy the Reduction object to our local stack frame
   // so that the compiler sees that it can hold the members in registers
   // avoiding writing back to memory after every loop step
   Reduction R1(R);

   const int innerLoopNum = e.shape()->length(1);
   int n = e.shape()->nelements();
   bool loop = true;

   if( e.isStride1() )
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating full reduction with stride 1\n";
#endif

      while( n && loop )
      {
         int j=0;
         for( ; j<innerLoopNum; ++j )
            if( !R1.evaluate( e.readWithoutStride(j) ) )
            {
               loop = false;
               break;
            }

         e.advance( innerLoopNum );
         e.advanceDim();
         n -= innerLoopNum;
      }
   }
   else
   {
#ifdef LTL_DEBUG_EXPRESSIONS
      cerr << "evaluating without common stride\n";
#endif

      while( n && loop )
      {
         int j=0;
         for( ; j<innerLoopNum; ++j )
            if( !R1.evaluate( e.readWithStride(j) ) )
            {
               loop = false;
               break;
            }
         e.advance( innerLoopNum );
         e.advanceDim();
         n -= innerLoopNum;
      }
   }
   R.copyResult( R1 );
}



// if we have index expressions involved, we cannot use the above
// optimizations, since we need to keep track of where we are
//
template<int N, class E, class Reduction>
void eval_full_reduction_with_index( ExprNode<E,N> e, Reduction& R )
{
#ifdef LTL_DEBUG_EXPRESSIONS
   cerr << "evaluating with pure stack traversal due to IndexIter\n";
#endif

   // copy the Reduction object to our local stack frame
   // so that the compiler sees that it can hold the members in registers
   // avoiding writing back to memory after every loop step
   Reduction R1(R);

   // we have an index iterator in the expression,
   // so we cannot do any loop unrolling or the like ...
   ShapeIter<N> i( *e.shape() );

   if( e.isStride1() )
   {
      while( !i.done() )
      {
         if( !R1.evaluate( *e ) )
            break;

         i.advanceWithStride1();
         e.advanceWithStride1();
         if( i.needAdvanceDim() )
         {
            i.advanceDim();
            e.advanceDim();
         }
      }
   }
   else
   {
      while( !i.done() )
      {
         if( !R1.evaluate( *e ) )
            break;

         i.advance();
         e.advance();
         if( i.needAdvanceDim() )
         {
            i.advanceDim();
            e.advanceDim();
         }
      }
   }
   R.copyResult( R1 );
}

}

#endif

