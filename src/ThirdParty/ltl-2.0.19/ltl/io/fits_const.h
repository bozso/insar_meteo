/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_const.h 513 2013-02-04 15:58:05Z cag $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#ifndef __FITS_CONST_H__
#define __FITS_CONST_H__

#include <ltl/config.h>
#include <cstddef>

namespace ltl {

//! Constants defining the physical geometry of FITS header.
typedef struct FitsPhysical_
{
      static const unsigned int RECORD_LENGTH;
      static const unsigned int CARD_LENGTH;
      static const unsigned int CARDS_OF_RECORD_LENGTH;
      static const unsigned int KEYWORD_LENGTH;
      static const unsigned int FIXED_VALUE_END;
      static const unsigned int FIXED_STRING_MIN_LENGTH;
      static const unsigned int FIXED_STRING_MIN_END;
      static const unsigned int VALUE_MIN_START;
      static const unsigned int COMMENT_MAX_LENGTH;
} FitsPhysical;

/*! 
  All reserved keywords must be stated here
  and in the static const structure holding
  the names and type information.
  To have easy loops always let SIMPLE at the beginning
  and END at the end, if you want to use the structure
  by yourself.
*/
typedef enum {
   SIMPLE, // SIMPLE must always be the first!

   BITPIX,
   NAXISxxx, // all NAXIS keywords
   XTENSION,
   PCOUNT,
   GCOUNT,
   TFIELDS,
   TFORMxxx,
   EXTEND,
   DATExxxx, // all DATE keywords
   ORIGIN,
   BLOCKED,
   TELESCOP,
   INSTRUME,
   OBSERVER,
   OBJECT,
   EQUINOXx,
   EPOCH,
   AUTHOR,
   REFERENC,
   COMMENT,
   HISTORY,
   Blank___, // keyword field is left blank
   BSCALE,
   BZERO,
   BUNIT,
   BLANK, // BLANK value for integer arrays
   DATAMAX,
   DATAMIN,
   WCSAXESx, // all keywords beginning with ...
   CRVALxxx, // ...
   CRPIXxxx,
   CDELTxxx,
   CROTAxxx,
   CTYPExxx,
   CUNITxxx,
   PCxx_xxx,
   CDxx_xxx,
   PVxx_xxx,
   PSxx_xxx,
   WCSNAMEx,
   CRDERxxx,
   CSYERxxx,
   LONPOLEx,
   LATPOLEx,
   RADESYSx,
   MJD_OBS,
   EXTNAME,
   EXTVER,
   EXTLEVEL,
   TTYPExxx,
   TUNITxxx,
   TNULLxxx,
   TSCALxxx,
   TZEROxxx,
   TDISPxxx,
   THEAP,
   TDIMxxxx,

   END, // end must always be the last!

   none // if key is not reserved
}reserved_keys_ident;

//! Identifiers for the function of a reserved key.
typedef enum {
   MANDATORY,
   ARRAY,
   DATE,
   COMMENTARY,
   EXTENSION,
   OTHER
}reserved_function_ident;

//! Identifiers for type of a reserved key.
typedef enum {
   EMPTY,
   STRING,
   BOOL,
   INT,
   FLOAT,
   COMPLEX
}reserved_type_ident;

//! This structure gives the syntax of a single reserved keyword.
typedef struct FitsKeyType_
{
//      enum { static_size = 3 * sizeof(int) + 9 * sizeof(char) + sizeof(bool)};
      const reserved_keys_ident IDENT;
      const reserved_function_ident FUNCTION;
      const reserved_type_ident TYPE;
      const char WORD [9];
      const bool FIXED;
} FitsKeyType;

//! Provides iterators through all known FITS keywords.
class KeyIterConst
{
   public:
      KeyIterConst(const FitsKeyType* ptr )
         : data_( ptr )
      { }
      
      inline FitsKeyType operator*() const
      {
         return *data_;
      }
      
      inline KeyIterConst& operator++()
      {
         ++data_;
         return *this;
      }
      
      inline KeyIterConst operator+(const int offset)
      {
         return KeyIterConst(data_ + offset);
      }
      
      inline KeyIterConst& operator--()
      {
         --data_;
         return *this;
      }

      inline bool operator==( const KeyIterConst& other ) const
      {
         return data_ == other.data_;
      }

      inline bool operator!=( const KeyIterConst& other ) const
      {
         return !(*this == other);
      }

   protected:
      FitsKeyType const * restrict_ data_;
};


/*! 
  Holds the list of all known reserved keywords
  and additional limits for the syntax check.
  The FitsKeyType array is implemented in a static const initializer
  in fits_const.cpp.
*/
class FitsKnownList
{
   public:
      typedef KeyIterConst const_iterator;
      
      static KeyIterConst begin();
      static KeyIterConst end();
      
      static const size_t N_KNOWN;
      static const FitsKeyType KEY[];
      // additional mandatory keywords limits
      static const int BITPIX_CHAR;
      static const int BITPIX_SHORT;
      static const int BITPIX_INT;
      static const int BITPIX_FLOAT;
      static const int BITPIX_DOUBLE;
      static const int NAXIS_LO_LIMIT;
      static const int NAXIS_HI_LIMIT;
};

}

#endif
