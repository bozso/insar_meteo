/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: config.h 497 2011-11-24 20:47:31Z drory $
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


#ifndef __LTL_CONFIG__
#define __LTL_CONFIG__


// read autoconf output
#include <ltl/acconfig.h>

// ====================================================================

// DEFAULT BEHAVIOUR : edit these to reflect your needs
//                     (can also be set manually before includig ltl
//                     headers. set reasonable default values here)

// ====================================================================

// range checking and behaviour on range errors
//

// range checking off by default
// #define LTL_RANGE_CHECKING
//
// default is to abort on range error
// #define LTL_ABORT_ON_RANGE_ERROR
// #define LTL_THROW_ON_RANGE_ERROR
//
// #undef LTL_DEBUG_EXPRESSIONS
// #undef LTL_DEBUG_MEMORY_BLOCK

//
// Thread safety
//
// Should we guard reference counting with mutex?
// This needs to be defined when running in a multithreaded
// setting, e.g. using OpenMP.
#ifdef LTL_MULTITHREAD
#  define LTL_THREADSAFE_MEMBLOCK
#endif

//
// Handling of loops in expression templates
//
// Should loops be unrolled by the LTL (see ltl/marray/eval.h)
//
#ifndef LTL_DONT_UNROLL_EXPRESSIONS
#  define LTL_UNROLL_EXPRESSIONS
#endif
// And for vectorized loops
#ifndef LTL_DONT_UNROLL_EXPRESSIONS_SIMD
#  define LTL_UNROLL_EXPRESSIONS_SIMD
#endif
//
// limit for loop unrolling in FVector and FMatrix classes
//
#ifndef LTL_TEMPLATE_LOOP_LIMIT
#  define LTL_TEMPLATE_LOOP_LIMIT 16
#endif

//
// vectorize loops using Altivec (PPC) or SSE2+ (i386)
//
//#  define LTL_USE_SIMD


// emit prefetch instructions
//
//#define LTL_EMIT_PREFETCH_GCC

// use namespaces 
//
#ifndef HAVE_NAMESPACES
#  error "Compiler does not support namespaces"
#endif
#define LTL_USING_NAMESPACE
#define UTIL_USING_NAMESPACE

// ====================================================================

// END OF USER PART - the rest should not be changed

// ====================================================================

//
// complex math
//
#ifdef HAVE_COMPLEX
#  define LTL_COMPLEX_MATH
#endif

// exactly one of these two MUST be defined if we have range checking on
// either LTL_ABORT_ON_RANGE_ERROR or LTL_THROW_ON_RANGE_ERROR
//
#if defined(LTL_RANGE_CHECKING) && !defined(LTL_ABORT_ON_RANGE_ERROR) && !defined(LTL_THROW_ON_RANGE_ERROR)
#  define LTL_ABORT_ON_RANGE_ERROR
#endif


/* support for proposed 'restrict' keyword  */
#ifndef HAVE_NCEG_RESTRICT
#  define restrict_
#endif


#if defined(HAVE_IEEE_MATH) && !defined(__SUNPRO_CC) && !defined(__GNUC__)
#  ifndef _ALL_SOURCE
#    define _ALL_SOURCE
#  endif
#  ifndef _XOPEN_SOURCE
#    define _XOPEN_SOURCE
#  endif
#  ifndef _XOPEN_SOURCE_EXTENDED
#    define _XOPEN_SOURCE_EXTENDED 1
#  endif
#endif


#if defined( HAVE_GCC_PREFETCH_BUILTINS ) && defined( LTL_EMIT_PREFETCH_GCC )
#  define LTL_PREFETCH_R(addr) __builtin_prefetch((addr),0,1)
#  define LTL_PREFETCH_RW(addr) __builtin_prefetch((addr),1,1)
#else
#  define LTL_PREFETCH_R(addr)
#  define LTL_PREFETCH_RW(addr)
#endif

#endif //__LTL_CONFIG__
