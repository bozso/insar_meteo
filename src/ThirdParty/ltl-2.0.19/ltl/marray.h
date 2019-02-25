/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: marray.h 556 2015-03-09 16:25:40Z drory $
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


// ====================================================================

#ifndef __LTL_MARRAY__
#define __LTL_MARRAY__

// ====================================================================

// little trick to avoid files being included separetely
#define __LTL_IN_FILE_MARRAY__
// this is #undef'd at the end of this file ...

// ====================================================================

#include <ltl/config.h>
#include <ltl/misc/mdebug.h>

#include <cmath>
#ifdef LTL_COMPLEX_MATH
#  include <complex>
   using std::complex;
#endif

#include <iostream>
#include <list>


// mandatory includes with own namespace ltl declaration
// (so that they can be used independently of marray.h)

//#include <ltl/fvector.h>
#include <ltl/misc/exceptions.h>
#include <ltl/misc/memblock.h>
#include <ltl/misc/range.h>

// ====================================================================

using std::list;

#include <ltl/marray/shape.h>
#include <ltl/marray/slice.h>
#include <ltl/misc/staticinit.h>
#include <ltl/marray/simplevec.h>

// ====================================================================

#include <ltl/misc/applicops.h>

#ifdef LTL_USE_SIMD
#  ifdef __ALTIVEC__
#    include <ltl/misc/applicops_altivec.h>
#  else
#    ifdef __SSE2__
#      include <ltl/misc/applicops_sse.h>
#    else
#      error "Either __ALTIVEC__ (for PPC) or __SSE2__ (for x86) have to be defined."
#    endif
#  endif
#endif

#include <ltl/marray/expr_base.h>
#include <ltl/marray/marray_iter.h>
#include <ltl/marray/expr.h>
#include <ltl/marray/expr_ops.h>
#include <ltl/marray/eval.h>
#include <ltl/marray/index_iter.h>

#include <ltl/marray/merge.h>
#include <ltl/marray/where.h>
#include <ltl/marray/cast.h>
#include <ltl/marray/apply.h>

// ====================================================================

namespace ltl {

//! A dynamic N-dimensional array storing objects of type T.
/*! \ingroup marray_class
  \c MArrays feature subarrays (sometimes called views), slices,
  expression templated evaluation, and other features described
  below. The memory holding the actual data is reference counted, such
  that it is not freed until the last reference to the data by views or
  slices has been removed.
*/
template<typename T, int N>
class MArray : public ExprBase<MArray<T,N>, N>
{

   public:
      friend class MArrayIter<T,N>;
      friend class MArrayIterConst<T,N>;
      friend class ListInitializationSwitch<MArray<T,N> >;

      typedef MArrayIter<T,N>       Iterator;
      typedef MArrayIterConst<T,N>  ConstIterator;

      //! Index iterator for ltl::where() (indexing arbitrary subsets).
      typedef IndexIter<T,N>        IndexIterator;
      typedef FixedVector<int,N>    IndexV;
      typedef IndexList<N>          IndexSet;

      /*! \name STL definitions:
       */
      //@{
      typedef T                     value_type;
      typedef T&                    reference;
      typedef const T&              const_reference;
      typedef T*                    pointer;
      typedef const T*              const_pointer;
      typedef size_t                size_type;
      typedef MArrayIter<T,N>       iterator;
      typedef MArrayIterConst<T,N>  const_iterator;
      //@}


      // CONSTRUCTORS
      //! Construct an array without allocating memory.
      /*! This is useful for constructing \c MArrays
        whose values are to be read from a file,
        and thus the dimensions are not known yet.

        \warning <em>Use this with care!</em>
        Use this \c MArray only after being sure that
        \c ltl::MArray::realloc() \c or ltl::MArray::makeReference() has been called.
      */
      MArray()
         : memBlock_(NULL), data_(NULL)
      {
      }


      /*! \name Construct an (optionally mapped) array of rank N with ltl::Range size arguments.
       */
      //@{
      MArray( const Range& r1, const bool map = false )
            : shape_( r1 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const bool map = false )
            : shape_( r1, r2 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const Range& r3,
              const bool map = false )
            : shape_(r1,r2,r3)
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const bool map = false )
            : shape_(r1,r2,r3,r4)
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5, const bool map = false )
            : shape_(r1,r2,r3,r4,r5)
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5, const Range& r6,
              const bool map=false )
         : shape_(r1,r2,r3,r4,r5,r6)
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5, const Range& r6,
              const Range& r7, const bool map=false )
         : shape_(r1,r2,r3,r4,r5,r6,r7)
      {
         setupMemory( map );
      }

      //@}

      /*! \name Construct an array of rank N with int size arguments.
      Index ranges <tt>{1..ni, i=1..N}</tt>,
        \c int arguments, base defaults to 1.
      */
      //@{
      /*! \overload
       */
      MArray( const int r1, const bool map = false )
            : shape_( r1 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const bool map = false )
            : shape_( r1, r2 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const int r3, const bool map = false )
            : shape_( r1, r2, r3 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const int r3, const int r4,
              const bool map = false )
            : shape_( r1, r2, r3, r4 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const int r3, const int r4,
              const int r5, const bool map = false)
            : shape_( r1, r2, r3, r4, r5 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const int r3, const int r4,
              const int r5, const int r6, const bool map = false)
         : shape_( r1, r2, r3, r4, r5, r6 )
      {
         setupMemory( map );
      }

      /*! \overload
       */
      MArray( const int r1, const int r2, const int r3, const int r4,
              const int r5, const int r6, const int r7,
              const bool map = false)
         : shape_( r1, r2, r3, r4, r5, r6, r7 )
      {
         setupMemory( map );
      }
      //@}

      //! Copy constructor. <b>Only makes a reference !!!</b>
      MArray( const MArray<T, N>& other )
            : memBlock_(0)
      {
         makeReference( other );
      }

      /*! \name Constructors for pure subarrays (rank preserved)

        Other array's data is referenced, NOT copied
        ( use operator=() for copy)!
        Missing arguments are treated as Range::all() !
      */
      //@{
      /*! \overload
       */
      MArray( const MArray<T, N>& other, const Range& r0 )
            : memBlock_(0)
      {
         subarray( other, r0 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2 )
            : memBlock_(0)
      {
         subarray( other, r1, r2 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2, const Range& r3 )
            : memBlock_(0)
      {
         subarray( other, r1, r2, r3 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2, const Range& r3,
              const Range& r4 )
            : memBlock_(0)
      {
         subarray( other, r1, r2, r3, r4 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5 )
            : memBlock_(0)
      {
         subarray( other, r1, r2, r3, r4, r5 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5, const Range& r6 )
            : memBlock_(0)
      {
         subarray( other, r1, r2, r3, r4, r5, r6 );
      }

      /*! \overload
       */
      MArray( const MArray<T, N>& other,
              const Range& r1, const Range& r2, const Range& r3,
              const Range& r4, const Range& r5, const Range& r6,
              const Range& r7 )
            : memBlock_(0)
      {
         subarray( other, r1, r2, r3, r4, r5, r6, r7 );
      }
      //@}

      //! Constructor for mixed slicing (int, rank reducing) and Range arguments.
      /*! Other array's data is referenced, NOT copied
        ( use operator=() for copy)!
      */
      // not for public use, user operator()
      template<int N2, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7>
      MArray( const MArray<T,N2>& other,
              R1 r1, R2 r2, R3 r3, R4 r4, R5 r5, R6 r6, R7 r7 )
            : memBlock_(0)
      {
         slicearray( other, r1, r2, r3, r4, r5, r6, r7 ) ;
      }

      /*! \name Construct from preexisting data.

        The array \c dims[] holds the lengths of each dimension
        (The number of dimensions, and hence the length of \c dims[]
        is known from the template parameter \c N.)
        No checking is performed.
        The index ranges are
        \c {1..dims[i]} along each dimension \c i.
      */
      //@{
      //! Construct from allocated memory. Note that this memory will be freed when the \c MArray is destructed
      MArray( T *data, const int * dims );

      //! Construct from map file.
      MArray( const string filename, const int * dims );
      //@}

      //! Construct from shape.
      /*! Useful to construct an array having the same geometry as another array.
        If you have map == true and give a filename the memory map will be saved.
       */
      MArray( const Shape<N>* s,
              const bool map = false, const char * filename = NULL );

      //! Construct from an array-valued expression.
      /*! The new array is allocated with the shape of the expression
        and its elements are filled with the evaluated expression.
        If you have map == true and give a filename the memory map will be saved.
      */
      template<typename Expr>
      MArray( const ExprNode<Expr,N>& e,
              const bool map = false, const char * filename = NULL );


      //! Decrement reference count of memory block. If 0 delete.
      ~MArray()
      {
         if( memBlock_ )
            memBlock_->removeReference();
         memBlock_ = NULL;
         data_     = NULL;
      }


      //! Free associated memory.
      /*! Frees the memory used to store the data before the MArray goes
        out of scope (before the destructor is being called). Can help keeping
        code cleaner without using pointers to MArray.
      */
      void free()
      {
         if( memBlock_ )
            memBlock_->removeReference();
         memBlock_ = NULL;
         data_     = NULL;
      }

      //! \c true if MArray currenltly has have no memory allocated.
      bool empty() const
      {
         return (memBlock_ == NULL);
      }

      //! \name Operator=, other's data is copied, own data overwritten
      /*!
       *  Arrays have to have conformable shapes, no automatic resize takes
       *  place. Use realloc() instead!
       *  If the lhs ltl::MArray is \c empty(), i.e. no memory has been allocated
       *  then memory conforming to \c other's shape will be allocated.
       */
      //@{
      //! This \code operator=() \endcode overloads the default.
      MArray<T,N>& operator=( const MArray<T,N>& other )
      {
         copy( other );
         return *this;
      }

      //! This \code operator=() \endcode is more general than the default.
      template<typename T2>
      MArray<T,N>& operator=( const MArray<T2,N>& other )
      {
         copy( other );
         return *this;
      }
      //@}

      //! Assignment of an expression to a \c MArray
      /*!
       *  The \c MArray and the expression have to have conformable shapes, no automatic
       *  resize takes place. Use realloc() instead!
       *  If the lhs ltl::MArray is \c empty(), i.e. no memory has been allocated
       *  then memory conforming to the expression's shape will be allocated.
       */
      template<typename Expr>
      MArray<T,N>& operator=( const ExprNode<Expr,N>& e );

      //! Assigns \c x to all elements.
      /*! A bit more comlicated since we have to discriminate between
        \code A = 3; and A = 1, 2, 3, 4;\endcode
        which is done using ListInitializationSwitch which either calls
        ListInitializer or MArray::fill().
      */
      ListInitializationSwitch< MArray<T,N> > operator=( T x )
      {
         return ListInitializationSwitch< MArray<T,N> >( *this, x );
      }



      /*! \name Overloaded X= operators. There is a version for an
          \code MArray \endcode rhs, an expression rhs, and a literal rhs for each operator.
          To have a single implementation of mathematical operations for scalar and
          vectorized code (where the C language extensions do not define X= ), we
          transform the \code A x= E \endcode assignment into \code A = A x E \endcode
          and call \code operator= \endcode .
       */
      //@{
      template<typename T2>
      MArray<T,N>& operator+=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator-=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator*=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator/=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator&=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator|=( const MArray<T2,N>& a );
      template<typename T2>
      MArray<T,N>& operator^=( const MArray<T2,N>& a );

      template<typename Expr>
      MArray<T,N>& operator+=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator-=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator*=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator/=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator&=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator|=( const ExprNode<Expr,N>& e );
      template<typename Expr>
      MArray<T,N>& operator^=( const ExprNode<Expr,N>& e );

      MArray<T,N>& operator+=( const T a );
      MArray<T,N>& operator-=( const T a );
      MArray<T,N>& operator*=( const T a );
      MArray<T,N>& operator/=( const T a );
      MArray<T,N>& operator&=( const T a );
      MArray<T,N>& operator|=( const T a );
      MArray<T,N>& operator^=( const T a );
      //@}


      /*! \name Elementwise array access (indexing) via operator()

      \c int / ltl::FixedVector arguments, return (reference to) element.
      */
      //@{
      // Integer arguments, return (reference to) element.
      T  operator()( const int i1 ) const
      {
         ASSERT_DIM(1);
         CHECK_BOUNDS1(i1);
         return data_[i1*shape_.stride_[0]];
      }
      T&  operator()( const int i1 )
      {
         ASSERT_DIM(1);
         CHECK_BOUNDS1(i1);
         return data_[i1*shape_.stride_[0]];
      }

      T  operator()( const int i1, const int i2 ) const
      {
         ASSERT_DIM(2);
         CHECK_BOUNDS2(i1,i2);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1]];
      }
      T& operator()( const int i1, const int i2 )
      {
         ASSERT_DIM(2);
         CHECK_BOUNDS2(i1,i2);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1]];
      }

      T  operator()( const int i1, const int i2,
                     const int i3 ) const
      {
         ASSERT_DIM(3);
         CHECK_BOUNDS3(i1,i2,i3);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2]];
      }
      T& operator()( const int i1, const int i2,
                               const int i3 )
      {
         ASSERT_DIM(3);
         CHECK_BOUNDS3(i1,i2,i3);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2]];
      }

      T  operator()( const int i1, const int i2,
                     const int i3, const int i4 ) const
      {
         ASSERT_DIM(4);
         CHECK_BOUNDS4(i1,i2,i3,i4);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] ];
      }
      T& operator()( const int i1, const int i2,
                               const int i3, const int i4 )
      {
         ASSERT_DIM(4);
         CHECK_BOUNDS4(i1,i2,i3,i4);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] ];
      }

      T  operator()( const int i1, const int i2,
                     const int i3, const int i4,
                     const int i5 ) const
      {
         ASSERT_DIM(5);
         CHECK_BOUNDS5(i1,i2,i3,i4,i5);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] ];
      }
      T& operator()( const int i1, const int i2,
                               const int i3, const int i4,
                               const int i5 )
      {
         ASSERT_DIM(5);
         CHECK_BOUNDS5(i1,i2,i3,i4,i5);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] ];
      }

      T  operator()( const int i1, const int i2,
                     const int i3, const int i4,
                     const int i5, const int i6 ) const
      {
         ASSERT_DIM(6);
         CHECK_BOUNDS6(i1,i2,i3,i4,i5,i6);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] + i6*shape_.stride_[5] ];
      }
      T& operator()( const int i1, const int i2,
                               const int i3, const int i4,
                               const int i5, const int i6 )
      {
         ASSERT_DIM(6);
         CHECK_BOUNDS6(i1,i2,i3,i4,i5,i6);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] + i6*shape_.stride_[5] ];
      }

      T  operator()( const int i1, const int i2,
                     const int i3, const int i4,
                     const int i5, const int i6, const int i7 ) const
      {
         ASSERT_DIM(7);
         CHECK_BOUNDS7(i1,i2,i3,i4,i5,i6,7);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] + i6*shape_.stride_[5] +
                      i7*shape_.stride_[6] ];
      }
      T& operator()( const int i1, const int i2,
                               const int i3, const int i4,
                               const int i5, const int i6, const int i7 )
      {
         ASSERT_DIM(7);
         CHECK_BOUNDS7(i1,i2,i3,i4,i5,i6,i7);
         return data_[i1*shape_.stride_[0] + i2*shape_.stride_[1] +
                      i3*shape_.stride_[2] + i4*shape_.stride_[3] +
                      i5*shape_.stride_[4] + i6*shape_.stride_[5] +
                      i7*shape_.stride_[6] ];
      }

      // FixedVector argument, return (reference to) element
      T operator()( const FixedVector<int,1>& i ) const
      {
         ASSERT_DIM(1);
         CHECK_BOUNDS1(i(1));
         return data_[i(1)*shape_.stride_[0]];
      }

      T& operator()( const FixedVector<int,1>& i )
      {
         ASSERT_DIM(1);
         CHECK_BOUNDS1(i(1));
         return data_[i(1)*shape_.stride_[0]];
      }

      T operator()( const FixedVector<int,2>& i ) const
      {
         ASSERT_DIM(2);
         CHECK_BOUNDS2(i(1),i(2));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]];
      }

      T& operator()( const FixedVector<int,2>& i )
      {
         ASSERT_DIM(2);
         CHECK_BOUNDS2(i(1),i(2));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]];
      }

      T operator()( const FixedVector<int,3>& i ) const
      {
         ASSERT_DIM(3);
         CHECK_BOUNDS3(i(1),i(2),i(3));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]];
      }

      T& operator()( const FixedVector<int,3>& i )
      {
         ASSERT_DIM(3);
         CHECK_BOUNDS3(i(1),i(2),i(3));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]];
      }

      T operator()( const FixedVector<int,4>& i ) const
      {
         ASSERT_DIM(4);
         CHECK_BOUNDS4(i(1),i(2),i(3),i(4));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]];
      }

      T& operator()( const FixedVector<int,4>& i )
      {
         ASSERT_DIM(4);
         CHECK_BOUNDS4(i(1),i(2),i(3),i(4));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]];
      }

      T operator()( const FixedVector<int,5>& i ) const
      {
         ASSERT_DIM(5);
         CHECK_BOUNDS5(i(1),i(2),i(3),i(4),i(5));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]];
      }

      T& operator()( const FixedVector<int,5>& i )
      {
         ASSERT_DIM(5);
         CHECK_BOUNDS5(i(1),i(2),i(3),i(4),i(5));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]];
      }

      T operator()( const FixedVector<int,6>& i ) const
      {
         ASSERT_DIM(6);
         CHECK_BOUNDS6(i(1),i(2),i(3),i(4),i(5),i(6));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]+i(5)*shape_.stride_[5]];
      }

      T& operator()( const FixedVector<int,6>& i )
      {
         ASSERT_DIM(6);
         CHECK_BOUNDS6(i(1),i(2),i(3),i(4),i(5),i(6));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]+i(6)*shape_.stride_[5]];
      }

      T operator()( const FixedVector<int,7>& i ) const
      {
         ASSERT_DIM(7);
         CHECK_BOUNDS7(i(1),i(2),i(3),i(4),i(5),i(6),i(7));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]+i(5)*shape_.stride_[5]+
                      i(6)*shape_.stride_[6]];
      }

      T& operator()( const FixedVector<int,7>& i )
      {
         ASSERT_DIM(7);
         CHECK_BOUNDS7(i(1),i(2),i(3),i(4),i(5),i(6),i(7));
         return data_[i(1)*shape_.stride_[0]+i(2)*shape_.stride_[1]+
                      i(3)*shape_.stride_[2]+i(4)*shape_.stride_[3]+
                      i(5)*shape_.stride_[4]+i(6)*shape_.stride_[5]+
                      i(7)*shape_.stride_[6]];
      }
      //@}

      /*! \name 'Cutting' array access (indexing) via operator()

      ltl::Range arguments, return (reference to) element.
      */
      //@{
      // Range arguments: return subarray
      MArray<T,N> operator()( const Range& r1 ) const
      {
         ASSERT_DIM(1);
         CHECK_RANGE(r1,1);
         return MArray<T,N>( *this, r1 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2 ) const
      {
         ASSERT_DIM(2);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         return MArray<T,N>( *this, r1, r2 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2,
                              const Range& r3 ) const
      {
         ASSERT_DIM(3);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         CHECK_RANGE(r3,3);
         return MArray<T,N>( *this, r1, r2, r3 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2,
                              const Range& r3, const Range& r4 ) const
      {
         ASSERT_DIM(4);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         CHECK_RANGE(r3,3);
         CHECK_RANGE(r4,4);
         return MArray<T,N>( *this, r1, r2, r3, r4 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2,
                              const Range& r3, const Range& r4,
                              const Range& r5 ) const
      {
         ASSERT_DIM(5);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         CHECK_RANGE(r3,3);
         CHECK_RANGE(r4,4);
         CHECK_RANGE(r5,5);
         return MArray<T,N>( *this, r1, r2, r3, r4, r5 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2,
                              const Range& r3, const Range& r4,
                              const Range& r5, const Range& r6 ) const
      {
         ASSERT_DIM(6);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         CHECK_RANGE(r3,3);
         CHECK_RANGE(r4,4);
         CHECK_RANGE(r5,5);
         CHECK_RANGE(r6,6);
         return MArray<T,N>( *this, r1, r2, r3, r4, r5, r6 );
      }

      MArray<T,N> operator()( const Range& r1, const Range& r2,
                              const Range& r3, const Range& r4,
                              const Range& r5, const Range& r6,
                              const Range& r7 ) const
      {
         ASSERT_DIM(7);
         CHECK_RANGE(r1,1);
         CHECK_RANGE(r2,2);
         CHECK_RANGE(r3,3);
         CHECK_RANGE(r4,4);
         CHECK_RANGE(r5,5);
         CHECK_RANGE(r6,6);
         CHECK_RANGE(r7,7);
         return MArray<T,N>( *this, r1, r2, r3, r4, r5, r6, r7 );
      }

      // mixed integer and Range arguments, return subarray/slice
      template<typename T1, typename T2>
      typename SliceCounter<T,T1,T2>::MArraySlice
      operator()(T1 r1, T2 r2) const
      {
         return typename SliceCounter<T,T1,T2>::
            MArraySlice(*this, r1, r2,
                        NoArgument(), NoArgument(), NoArgument(), NoArgument(), NoArgument());
      }

      template<typename T1, typename T2, typename T3>
      typename SliceCounter<T,T1,T2,T3>::MArraySlice
      operator()(T1 r1, T2 r2, T3 r3) const
      {
         return typename SliceCounter<T,T1,T2,T3>::
            MArraySlice(*this, r1, r2, r3, NoArgument(), NoArgument(), NoArgument(), NoArgument() );
      }

      template<typename T1, typename T2, typename T3, typename T4>
      typename SliceCounter<T,T1,T2,T3,T4>::MArraySlice
      operator()(T1 r1, T2 r2, T3 r3, T4 r4) const
      {
         return typename SliceCounter<T,T1,T2,T3,T4>::
            MArraySlice(*this, r1, r2, r3, r4, NoArgument(), NoArgument(), NoArgument() );
      }

      template<typename T1, typename T2, typename T3, typename T4, typename T5>
      typename SliceCounter<T,T1,T2,T3,T4,T5>::MArraySlice
      operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5) const
      {
         return typename SliceCounter<T,T1,T2,T3,T4,T5>::
            MArraySlice(*this, r1, r2, r3, r4, r5, NoArgument(), NoArgument() );
      }

      template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
      typename SliceCounter<T,T1,T2,T3,T4,T5,T6>::MArraySlice
      operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6) const
      {
         return typename SliceCounter<T,T1,T2,T3,T4,T5,T6>::
            MArraySlice(*this, r1, r2, r3, r4, r5, r6, NoArgument() );
      }

      template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
      typename SliceCounter<T,T1,T2,T3,T4,T5,T6,T7>::MArraySlice
      operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7) const
      {
         return typename SliceCounter<T,T1,T2,T3,T4,T5,T6,T7>::
            MArraySlice(*this, r1, r2, r3, r4, r5, r6, r7 );
      }
      //@}

      //! Index with ltl::IndexList (rhs version).
      /*! Return ltl::MArray<T,1> object.
       */
      MArray<T,1> operator()( const IndexList<N>& l ) const
      {
         typename std::list<IndexV>::const_iterator i=l.begin(), e=l.end();
         MArray<T,1> v( l.size() );

         for( int j=1; i!=e; ++i, ++j )
            v(j) = (*this)( *i );

         return v;
      }

      //! Index with ltl::IndexList (lhs version).
      /*! Assignment to self indexed with IndexList A[l] = Expr<T,1>
        IndexRef object is needed to carry the operator=( Expr ) method.
      */
      IndexRef<T,N> operator[]( const IndexList<N>& l )
      {
         return IndexRef<T,N>( l, *this );
      }

      /*! \name Change bases:

      Change the bases of the array (the bases are the first indices of each
      dimension)
       */
      //@{
      void setBase( const int b1 );
      void setBase( const int b1, const int b2 );
      void setBase( const int b1, const int b2, const int b3 );
      void setBase( const int b1, const int b2, const int b3, const int b4 );
      void setBase( const int b1, const int b2, const int b3, const int b4,
                    const int b5 );
      void setBase( const int b1, const int b2, const int b3, const int b4,
                    const int b5, const int b6 );
      void setBase( const int b1, const int b2, const int b3, const int b4,
                    const int b5, const int b6, const int b7 );
      //@}

      //! Make this being a referece to \e other's data.
      void makeReference( const MArray<T,N>& other );

      //! Make a reference as a different-dimensional view of another \c MArray's data
      /*! The new array is referencing the \c other's data but takes it to have
       * rank \c N instead of N2 and dimensions \c dims[0] ... \c dims[N-1].
       * Use with care. Memory must be contiguous.
       */
      template<int N2>
      void makeReferenceWithDims( const MArray<T,N2>& other, const int* dims);


      //! Reallocate memory. Data are abolished.
      /*! If you have map == true and give a filename
        the new memory map will be saved.
       */
      void realloc( const Shape<N>& s,
                    const bool map = false, const char * filename = NULL );


      /* \name Iterators:

      These fulfill requirements of forward iterators at most ...
      */
      //@{
      iterator iter()
      {
         return iterator( *this );
      }

      iterator begin()
      {
         return iterator( *this );
      }

      const_iterator begin() const
      {
         return const_iterator( *this );
      }

      iterator end()
      {
         return iterator( *this, _iter_end_tag() );
      }

      const_iterator end() const
      {
         return const_iterator( *this, _iter_end_tag() );
      }

      /*! In case our memory layout is contiguous, we can offer an
         easy solution to provide random access iterators.
      */
      T* beginRA()
      {
         LTL_ASSERT( isStorageContiguous(),
                     "Can't construct random-access iterator on non contiguous MArray!" );
         return data();
      }

      const T* beginRA() const
      {
         LTL_ASSERT( isStorageContiguous(),
                     "Can't construct random-access iterator on non contiguous MArray!" );
         return data();
      }

      T* endRA()
      {
         return beginRA() + nelements();
      }

      const T* endRA() const
      {
         return beginRA() + nelements();
      }
      //@}

      //! Pointer to first data element.
      /*! \warning The second data element NEED NOT be at first + 1
        since the memeory might be non-contiguous. Use iteators!
      */
      T* data() const
      {
         return data_ - shape_.zeroOffset();
      }

      //! An iterator-style thing providing indices when dereferenced.
      /*! I.e. *i or i() gives a FixedVector holding the indices and
        and i(int) gives the index in the i-th dimension
      */
      IndexIterator indexBegin() const
      {
         return IndexIterator( *this );
      }


      /*! \name Array information:

      Interfaces ltl::Shape.
      */
      //@{
      //! Number of elements .
      size_type nelements() const
      {
         return shape_.nelements();
      }

      //! Number of elements .
      size_type size() const
      {
         return nelements();
      }

      //! First index along dimension \e dim (starting to count at 1)
      int minIndex( const int dim ) const
      {
         CHECK_DIM(dim);
         return shape_.base(dim);
      }

      //! Last index along dimension \e dim (starting to count at 1)
      int maxIndex( const int dim ) const
      {
         CHECK_DIM(dim);
         return shape_.last(dim);
      }

      //! Length of dimension \e dim.
      int length( const int dim ) const
      {
         CHECK_DIM(dim);
         return shape_.length(dim);
      }

      //! Stride of dimension \e dim.
      int stride( const int dim ) const
      {
         CHECK_DIM(dim);
         return shape_.stride(dim);
      }

      bool isStride1() const
      {
         return stride(1) == 1;
      }

      int zeroOffset() const
      {
         return shape_.zeroOffset();
      }

      bool isStorageContiguous() const
      {
         return shape_.isStorageContiguous();
      }

      //! Check conformability with \e other ltl::Shape.
      bool isConformable( const Shape<N>& other ) const
      {
         return shape_.isConformable( other );
      }

      //! Check conformability with \e other array.
      template<typename T2>
      bool isConformable( const MArray<T2,N>& other ) const
      {
         return shape_.isConformable( *other.shape() );
      }

      //! Return the \c ltl::Shape.
      const Shape<N>* shape() const
      {
         return &shape_;
      }
      //@}

      //! Return pointer to associated MemoryBlock .
      MemoryBlock<T>* memoryBlock() const
      {
         return memBlock_;
      }

      //! Return true, if \c MArray has associated MemoryBlock.
      bool isAllocated() const
      {
         return !empty();
      }

      /*! \name Reverse and Transpose:

        Reorder array (without copy). Very fast!
      */
      //@{
      //! Reverse this MArray.
      void reverseSelf( const int dim=1 );
      //! Return reversed MArray.
      MArray<T, N> reverse( const int dim=1 ) const;
      //! Transpose this MArray.
      void transposeSelf( const int dim1=1, const int dim2=2 );
      //! Return transposed MArray.
      MArray<T, N> transpose( const int dim1=1, const int dim2=2 ) const;
      //@}

      //! Debug output. Print geometry information.
      void describeSelf() const;


   protected:
      //! Copy from \e other.
      template<typename T2>
      void copy( const MArray<T2,N>& other );

      //! Fill with \e t.
      void fill( const T t );

      // SUBARRAYS
      //! Constructs a pure subarray of \e other, i.e. rank is preserved.
      void subarray( const MArray<T,N>& other,
                     const Range& r1 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2,
                     const Range& r3 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2,
                     const Range& r3, const Range& r4 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2,
                     const Range& r3, const Range& r4, const Range& r5 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2,
                     const Range& r3, const Range& r4, const Range& r5,
                     const Range& r6 );

      /*! \overload
       */
      void subarray( const MArray<T,N>& other,
                     const Range& r1, const Range& r2,
                     const Range& r3, const Range& r4, const Range& r5,
                     const Range& r6, const Range& r7 );


      // SLICING (changes rank -> can't be public, use consructors ...)
      template<int N2, typename R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7>
      void slicearray( const MArray<T,N2>& other,
                       R1 r1, R2 r2, R3 r3, R4 r4, R5 r5, R6 r6, R7 r7 );

      template<int N2>
      void slice( int& setRank, Range r, const MArray<T,N2>& other,
                  int sourceRank );

      template<int N2>
      void slice( int& setRank, int r, const MArray<T,N2>& other,
                  int sourceRank );

      template<int N2>
      void slice( int& /*setRank*/, NoArgument,
                  const MArray<T,N2>& /*other*/,
                  int /*sourceRank*/ )
      { }

      // INTERNAL UTIL METHODS
      void setupMemory(const bool map = false, const char * filename = NULL);

      void setupShape( const int * dims );

      void setrange( const int dim, const Range& r );


      // DATA MEMBERS

      //! Our MemoryBlock.
      MemoryBlock<T>* memBlock_;

      //! Holds shape information.
      /*! And zeroOffset_ which is the distance between data_ and the
        the address of the first element of the array, such that
        \code
        first = data_ + sum( base(i) * stride(i) )
              = data_ + shape_.zeroOffset();
        \endcode
        always holds.
      */
      Shape<N> shape_;
   public:
      //! Pointer to element ( 0, 0, ..., 0 )
      T* restrict_    data_;
};

// ====================================================================

}

#include <ltl/marray/marray_ops.h>
#include <ltl/marray/marray_methods.h>

#undef __LTL_IN_FILE_MARRAY__

#endif // __LTL_MARRAY__
