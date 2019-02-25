/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: type_promote.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __LTL_TYPE_PROMOTE__
#define __LTL_TYPE_PROMOTE__

#include <ltl/config.h>

namespace ltl {

/*! \file type_promote.h
  Templates for c-style type promotion (see Stroustrup, C.6.3, 3rd ed.)
*/

// first, we order types according to their 'precision'
// ASSUME THAT NON-BUILTIN TYPES HAVE HIGHEST PRECISION!
template<class T>
struct precision_trait
{
   enum { precision = 10000 };
};


#define LTL_TYPE_PRECISION(T,p)                 \
template<>                                      \
struct precision_trait< T >                     \
{                                               \
      enum { precision = p };                   \
};


// note that smaller integral types than int get promoted to int
// anyway according to the standard, that's why they are missing here,
// see below ...
LTL_TYPE_PRECISION(int,10)
LTL_TYPE_PRECISION(unsigned int,20)
LTL_TYPE_PRECISION(long,30)
LTL_TYPE_PRECISION(unsigned long,40)
LTL_TYPE_PRECISION(float,50)
LTL_TYPE_PRECISION(double,60)
#ifdef HAVE_LONG_DOUBLE
  LTL_TYPE_PRECISION(long double,70)
#endif

#ifdef HAVE_COMPLEX
LTL_TYPE_PRECISION(complex<float>,80)
LTL_TYPE_PRECISION(complex<double>,90)
#  ifdef HAVE_LONG_DOUBLE
      LTL_TYPE_PRECISION(complex<long double>,100)
#  endif
#endif

// now we handle promotion of small integral types to int/unsigned int
template<class T>
struct promote_smallint
{
   typedef T PType;
};


#define LTL_PROMOTE_SMALLINT(T1,T2)             \
template<>                                      \
struct promote_smallint<T1>                     \
{                                               \
      typedef T2 PType;                         \
};


LTL_PROMOTE_SMALLINT(bool,int)
LTL_PROMOTE_SMALLINT(char,int)
LTL_PROMOTE_SMALLINT(unsigned char,int)
LTL_PROMOTE_SMALLINT(short int,int)
LTL_PROMOTE_SMALLINT(short unsigned int,unsigned int)


// promote to T2 if which is !=0, promote to T1 otherwise
template<class T1, class T2, int which>
struct promote_to
{
   typedef T2 PType;
};


template<class T1, class T2>
struct promote_to<T1,T2,0>
{
   typedef T1 PType;
};



// now we can define the template class that actually handles
// type promotion

template<class T1, class T2>
struct promotion_trait
{
   // promote small integers to int/unsigned int
   typedef typename promote_smallint<T1>::PType T_1;
   typedef typename promote_smallint<T2>::PType T_2;

   // promote to highest precision
   enum { T2_has_higher_precision =
             (int)(precision_trait<T_1>::precision) <
             (int)(precision_trait<T_2>::precision) };

   typedef typename
   promote_to<T_1, T_2, T2_has_higher_precision>::PType PType;
};


// =====================================================================

// type promotion for sums and products

#define LTL_PROMOTE_SUM(Type,Sum_Type)          \
template<>                                      \
struct sumtype_trait<Type>                      \
{                                               \
      typedef Sum_Type SumType;                 \
};


template<class Type>
struct sumtype_trait
{
      
};


LTL_PROMOTE_SUM(char,int)
LTL_PROMOTE_SUM(short,int)
LTL_PROMOTE_SUM(int,long)
LTL_PROMOTE_SUM(long,long)
LTL_PROMOTE_SUM(unsigned char, unsigned int)
LTL_PROMOTE_SUM(unsigned short,unsigned int)
LTL_PROMOTE_SUM(unsigned int,  unsigned long)
LTL_PROMOTE_SUM(unsigned long, unsigned long)
LTL_PROMOTE_SUM(float,double)
LTL_PROMOTE_SUM(double,double)
#ifdef HAVE_LONG_DOUBLE
LTL_PROMOTE_SUM(long double,long double)
#endif

}

#endif // __LTL_TYPE_PROMOTE__

