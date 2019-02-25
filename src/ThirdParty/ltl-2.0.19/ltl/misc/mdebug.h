/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: mdebug.h 543 2014-07-09 22:56:36Z drory $
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


#if !defined(__LTL_IN_FILE_MARRAY__) && !defined(__LTL_IN_FILE_FVECTOR__) && !defined(__LTL_IN_FILE_FMATRIX__)
#error "<ltl/misc/mdebug.h> must be included via ltl headers, never alone!"
#endif


#ifndef __LTL_DEBUG__
#define __LTL_DEBUG__

#include <ltl/config.h>

#include <cstdlib>
#include <iostream>

// silence unused variable warnings:
#define LTL_UNUSED(x) ((void)(x))

#ifdef LTL_THROW_ON_RANGE_ERROR
#include <ltl/misc/exceptions.h>
#include <string>
#include <sstream>
using std::string;
using std::stringstream;
#endif

using std::cerr;
using std::cout;
using std::endl;


#ifndef __LINE__
#define __LINE__ "unknown"
#endif

#ifndef __FILE__
#define __FILE__ "unknown"
#endif

#ifdef __GNUC__
#define __LTL_FILE_LINE \
  "file '"<<__FILE__<<"', line "<<__LINE__<<",\nfunction "<<__PRETTY_FUNCTION__
#else
#define __LTL_FILE_LINE \
  "file '"<<__FILE__<<"', line "<<__LINE__
#endif


#define LTL_EXPECT_(a,b,m)        \
if ( (a) != (b) )                 \
{                                 \
   cerr<<"Expected " << a << ", got " << b << endl;         \
   cerr<<"Assertion failed in "<<__LTL_FILE_LINE<<" :\n";   \
   cerr<<m;                       \
   cerr<<endl;                    \
   cerr.flush();                  \
   abort();                       \
}


// this is for assertions (mostly within expression templates
// and internal consistency checks ...)
// always forces abort()
//
#define LTL_ASSERT_( x, m )					\
   if (!(x))							\
   {								\
      cerr<<"Assertion failed in "<<__LTL_FILE_LINE<<" :\n";	\
      cerr<<m;							\
      cerr<<endl;						\
      cerr.flush();						\
      abort();							\
   }


#ifdef LTL_RANGE_CHECKING
// on range checking errors we can either abort() and core-dump or
// throw an exception:
#if !defined(LTL_ABORT_ON_RANGE_ERROR) && !defined(LTL_THROW_ON_RANGE_ERROR)
  #error "Either LTL_ABORT_ON_RANGE_ERROR or LTL_THROW_ON_RANGE_ERROR must be defined!"
#endif

#if defined(LTL_ABORT_ON_RANGE_ERROR) && defined(LTL_THROW_ON_RANGE_ERROR)
  #error "LTL_ABORT_ON_RANGE_ERROR and LTL_THROW_ON_RANGE_ERROR cannot both be defined!"
#endif
#endif



// the next two are for range checking.
// either abort() or throw an exception ...
//
#ifdef LTL_ABORT_ON_RANGE_ERROR
#define LTL_RCHECK( x, m )					\
   if (!(x))							\
   {								\
      cerr<<"Range check failed in "<<__LTL_FILE_LINE<<" :\n";	\
      cerr<<m;							\
      cerr<<endl;						\
      cerr.flush();						\
      abort();							\
   }
#endif

#ifdef LTL_THROW_ON_RANGE_ERROR
#define LTL_RCHECK( x, m )						\
   if (!(x))								\
   {									\
      stringstream __ltl_errstr;					\
      __ltl_errstr<<"Range check failed in "<<__LTL_FILE_LINE<<" :\n";	\
      __ltl_errstr<<m<<endl;					        \
      string __ltl_errstring = __ltl_errstr.str();			\
      throw RangeException( __ltl_errstring );				\
   }
#endif


// OK, now define or undefine the user-level macros accordingly
#ifdef LTL_RANGE_CHECKING

#define LTL_ASSERT(x,m) LTL_ASSERT_(x,m)

#define ASSERT_DIM(d)  					\
   LTL_ASSERT( N==d,"MArray not of dimension "<<d	\
	       << shape_ )

#define CHECK_DIM(d)                                                    \
   LTL_ASSERT((d)>0 && (d)<=N,                                          \
	      "Illegal dimension: "<<d<<" for MArray<"<<N<<">"          \
	      << shape_ )

#define CHECK_SLICE(s,d) 						\
   LTL_ASSERT(unsigned(s-other.minIndex(d))< unsigned(other.length(d)),	\
	      "Bad Slice("<<d<<","<<s<<")\n"				\
	      << *other.shape() )


#define CHECK_CONFORM(expr,array) 				\
   LTL_ASSERT( (expr).isConformable(*((array).shape())),	\
	       "Operands not conformable in expression")

#define CHECK_CONFORM_MSG(expr,array,m) 			\
   LTL_ASSERT( (expr).isConformable(*((array).shape())),	\
	       "Operands not conformable in operator "<<m )


#define CHECK_RANGE(r,d)						 \
    LTL_RCHECK((r.first()==minStart && r.last()==minEnd) ||              \
              (((r.first()<=r.last() && r.stride()>0) ||                 \
	       (r.first()>=r.last() && r.stride()<0)) &&                 \
              (unsigned(r.first()-minIndex(d)) < unsigned(length(d)) &&  \
               unsigned(r.last()-minIndex(d))  < unsigned(length(d)))),  \
	      "Bad Range("<<r.first()<<","<<r.last()<<") for dimension " \
		 <<d<<endl<<shape_ )


#define CHECK_BOUNDS1(i1) 					\
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)),	\
	       "Index out of bounds: ("<<i1<<")\n"		\
	       <<shape_ )

#define CHECK_BOUNDS2(i1,i2) 					\
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<")\n"	\
	       <<shape_ )

#define CHECK_BOUNDS3(i1,i2,i3) 				\
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)) &&	\
	       unsigned(i3-minIndex(3))<unsigned(length(3)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<","	\
	       <<i3<<")\n"					\
	       <<shape_ )

#define CHECK_BOUNDS4(i1,i2,i3,i4) 				\
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)) &&	\
	       unsigned(i3-minIndex(3))<unsigned(length(3)) &&	\
	       unsigned(i4-minIndex(4))<unsigned(length(4)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<","	\
	       <<i3<<","<<i4<<")\n"				\
	       <<shape_ )

#define CHECK_BOUNDS5(i1,i2,i3,i4,i5) 				\
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)) &&	\
	       unsigned(i3-minIndex(3))<unsigned(length(3)) &&	\
	       unsigned(i4-minIndex(4))<unsigned(length(4)) &&	\
	       unsigned(i5-minIndex(5))<unsigned(length(5)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<","	\
	       <<i3<<","<<i4<<","<<i5<<")\n"			\
	       <<shape_ )

#define CHECK_BOUNDS6(i1,i2,i3,i4,i5,i6)                        \
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)) &&	\
	       unsigned(i3-minIndex(3))<unsigned(length(3)) &&	\
	       unsigned(i4-minIndex(4))<unsigned(length(4)) &&	\
	       unsigned(i5-minIndex(5))<unsigned(length(5)) &&	\
	       unsigned(i6-minIndex(6))<unsigned(length(6)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<","	\
	       <<i3<<","<<i4<<","<<i5<<","<<i6<<")\n"           \
	       <<shape_ )

#define CHECK_BOUNDS7(i1,i2,i3,i4,i5,i6,i7)                     \
   LTL_RCHECK( unsigned(i1-minIndex(1))<unsigned(length(1)) &&	\
	       unsigned(i2-minIndex(2))<unsigned(length(2)) &&	\
	       unsigned(i3-minIndex(3))<unsigned(length(3)) &&	\
	       unsigned(i4-minIndex(4))<unsigned(length(4)) &&	\
	       unsigned(i5-minIndex(5))<unsigned(length(5)) &&	\
	       unsigned(i6-minIndex(6))<unsigned(length(6)) &&	\
	       unsigned(i7-minIndex(7))<unsigned(length(7)),	\
	       "Index out of bounds: ("<<i1<<","<<i2<<","	\
	       <<i3<<","<<i4<<","<<i5<<","<<i6<<","<<i7<<")\n"  \
	       <<shape_ )

#else

#define LTL_ASSERT(x,m)
#define ASSERT_DIM(d)      LTL_UNUSED(d)
#define CHECK_DIM(x)       LTL_UNUSED(x)
#define CHECK_SLICE(s,d)

#define CHECK_CONFORM(shape1,shape2)
#define CHECK_CONFORM_MSG(shape1,shape2,m)

#define CHECK_RANGE(r,d)

#define CHECK_BOUNDS1(i1)
#define CHECK_BOUNDS2(i1,i2)
#define CHECK_BOUNDS3(i1,i2,i3)
#define CHECK_BOUNDS4(i1,i2,i3,i4)
#define CHECK_BOUNDS5(i1,i2,i3,i4,i5)
#define CHECK_BOUNDS6(i1,i2,i3,i4,i5,i6)
#define CHECK_BOUNDS7(i1,i2,i3,i4,i5,i6,i7)

#endif

#endif // __LTL_DEBUG__

