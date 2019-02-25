/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: swapbytes.h 495 2011-11-06 23:44:39Z drory $
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




#ifndef __LTL_BYTESWAP__
#define __LTL_BYTESWAP__



#include <ltl/config.h>



// Byte swapping stuff for FITS I/O
//

#ifdef WORDS_BIGENDIAN
// Nothing to be done, we are already big-endian

#  define LTL_TO_BIGENDIAN_16(val)   (val)
#  define LTL_TO_BIGENDIAN_32(val)   (val)
#  define LTL_TO_BIGENDIAN_64(val)   (val)
#  define LTL_FROM_BIGENDIAN_16(val) (val)
#  define LTL_FROM_BIGENDIAN_32(val) (val)
#  define LTL_FROM_BIGENDIAN_64(val) (val)

#else
// Must swap, we are little-endian

// 16-bit swap
#  define LTL_UINT16_SWAP_(val)        ((uint16_t) (    \
    (((uint16_t) (val) & (uint16_t) 0x00ffU) << 8) |    \
    (((uint16_t) (val) & (uint16_t) 0xff00U) >> 8)))

#  ifdef HAVE_BSWAP_BUILTINS
// Compiler supports builtin functions for 32 and 64 bits, use them!
#  define LTL_UINT32_SWAP_(val)    (__builtin_bswap32(val))
#  define LTL_UINT64_SWAP_(val)    (__builtin_bswap64(val))

#  else
// Implement 32 and 64 bit swap
#  define LTL_UINT32_SWAP_(val)        ((uint32_t) (            \
    (((uint32_t) (val) & (uint32_t) 0x000000ffU) << 24) |       \
    (((uint32_t) (val) & (uint32_t) 0x0000ff00U) <<  8) |       \
    (((uint32_t) (val) & (uint32_t) 0x00ff0000U) >>  8) |       \
    (((uint32_t) (val) & (uint32_t) 0xff000000U) >> 24)))

#  define LTL_UINT64_SWAP_(val)       ((uint64_t) (       \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x00000000000000ffULL) << 56) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x000000000000ff00ULL) << 40) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x0000000000ff0000ULL) << 24) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x00000000ff000000ULL) <<  8) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x000000ff00000000ULL) >>  8) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x0000ff0000000000ULL) >> 24) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0x00ff000000000000ULL) >> 40) |        \
      (((uint64_t) (val) &                                \
        (uint64_t) 0xff00000000000000ULL) >> 56)))
#  endif // HAVE_BSWAP_BUILTINS

#  define LTL_TO_BIGENDIAN_16(val)   LTL_UINT16_SWAP_(val)
#  define LTL_TO_BIGENDIAN_32(val)   LTL_UINT32_SWAP_(val)
#  define LTL_TO_BIGENDIAN_64(val)   LTL_UINT64_SWAP_(val)
#  define LTL_FROM_BIGENDIAN_16(val) LTL_UINT16_SWAP_(val)
#  define LTL_FROM_BIGENDIAN_32(val) LTL_UINT32_SWAP_(val)
#  define LTL_FROM_BIGENDIAN_64(val) LTL_UINT64_SWAP_(val)

#endif  // WORDS_BIGENDIAN

#endif // __LTL_BYTESWAP__
