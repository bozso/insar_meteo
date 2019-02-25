/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_const.cpp 513 2013-02-04 15:58:05Z cag $
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

#include <io/fits_const.h>

namespace ltl {

// FitsPhysical constants
const unsigned int FitsPhysical::RECORD_LENGTH = 2880u;
const unsigned int FitsPhysical::CARD_LENGTH = 80u;
const unsigned int FitsPhysical::CARDS_OF_RECORD_LENGTH = 36u;
const unsigned int FitsPhysical::KEYWORD_LENGTH = 8u;
const unsigned int FitsPhysical::FIXED_VALUE_END = 30u;
const unsigned int FitsPhysical::FIXED_STRING_MIN_LENGTH = 8u;
const unsigned int FitsPhysical::FIXED_STRING_MIN_END = 19u;
const unsigned int FitsPhysical::VALUE_MIN_START = 10u;
const unsigned int FitsPhysical::COMMENT_MAX_LENGTH = 71u;


// ------------------------------------------------------------------------
// iterators through known list
KeyIterConst FitsKnownList::begin()
{
   return KeyIterConst(KEY);
}
KeyIterConst FitsKnownList::end()
{
   return begin() + N_KNOWN;
}

// ------------------------------------------------------------------------
//! Number of known keywords.
   const size_t FitsKnownList::N_KNOWN = reserved_keys_ident(none);
//! List of known keywords.
const FitsKeyType FitsKnownList::KEY[] =
{
   {SIMPLE  , MANDATORY,  BOOL,    "SIMPLE  ", true}, //  0, start here

   {BITPIX  , MANDATORY,  INT,     "BITPIX  ", true},
   {NAXISxxx, MANDATORY,  INT,     "NAXIS"   , true}, // (max. 1000 NAXIS words)

   {XTENSION, MANDATORY,  STRING,  "XTENSION", true}, 
   {PCOUNT  , EXTENSION,  INT,     "PCOUNT  ", true}, // "mandatory" keys for
   {GCOUNT  , EXTENSION,  INT,     "GCOUNT  ", true}, // specific extensions
   {TFIELDS , EXTENSION,  INT,     "TFIELDS ", true},
   {TFORMxxx, EXTENSION,  STRING,  "TFORM"   , false},      
   {EXTEND  , EXTENSION,  BOOL,    "EXTEND  ", true},

   {DATExxxx, DATE,       STRING,  "DATE",     false}, // any DATE format keyword

   {ORIGIN  , OTHER,      STRING,  "ORIGIN  ", false},
   {BLOCKED , OTHER,      BOOL,    "BLOCKED ", false},
   {TELESCOP, OTHER,      STRING,  "TELESCOP", false},
   {INSTRUME, OTHER,      STRING,  "INSTRUME", false},
   {OBSERVER, OTHER,      STRING,  "OBSERVER", false},
   {OBJECT  , OTHER,      STRING,  "OBJECT  ", false},
   {EQUINOXx, OTHER,      FLOAT,   "EQUINOX", false},
   {EPOCH   , OTHER,      FLOAT,   "EPOCH   ", false},
   {AUTHOR  , OTHER,      STRING,  "AUTHOR  ", false},
   {REFERENC, OTHER,      STRING,  "REFERENC", false},

   {COMMENT , COMMENTARY, EMPTY,   "COMMENT ", false},
   {HISTORY , COMMENTARY, EMPTY,   "HISTORY ", false},
   {Blank___, COMMENTARY, EMPTY,   "        ", false},

   {BSCALE  , ARRAY,      FLOAT,   "BSCALE  ", false},
   {BZERO   , ARRAY,      FLOAT,   "BZERO   ", false},
   {BUNIT   , ARRAY,      STRING,  "BUNIT   ", false},
   {BLANK   , ARRAY,      INT,     "BLANK   ", false},
   {DATAMAX , ARRAY,      FLOAT,   "DATAMAX ", false},
   {DATAMIN , ARRAY,      FLOAT,   "DATAMIN ", false},
   {WCSAXESx, ARRAY,      INT,     "WCSAXES",  false},
   {CRVALxxx, ARRAY,      FLOAT,   "CRVAL",    false},
   {CRPIXxxx, ARRAY,      FLOAT,   "CRPIX",    false},
   {CDELTxxx, ARRAY,      FLOAT,   "CDELT",    false},
   {CROTAxxx, ARRAY,      FLOAT,   "CROTA",    false},
   {CTYPExxx, ARRAY,      STRING,  "CTYPE",    false},
   {CUNITxxx, ARRAY,      STRING,  "CUNIT",    false},
   {PCxx_xxx, ARRAY,      FLOAT,   "PC",       false},
   {CDxx_xxx, ARRAY,      FLOAT,   "CD",       false},
   {PVxx_xxx, ARRAY,      FLOAT,   "PV",       false},
   {PSxx_xxx, ARRAY,      STRING,  "PS",       false},
   {WCSNAMEx, ARRAY,      STRING,  "WCSNAME",  false},
   {CRDERxxx, ARRAY,      FLOAT,   "CRDER",    false},
   {CSYERxxx, ARRAY,      FLOAT,   "CSYER",    false},
   {LONPOLEx, ARRAY,      FLOAT,   "LONPOLE",  false},
   {LATPOLEx, ARRAY,      FLOAT,   "LATPOLE",  false},
   {RADESYSx, ARRAY,      STRING,  "RADESYS",  false},
   {MJD_OBS , ARRAY,      FLOAT,   "MJD-OBS ", false},

   {EXTNAME , EXTENSION,  STRING,  "EXTNAME ", false},
   {EXTVER  , EXTENSION,  INT,     "EXTVER  ", false},
   {EXTLEVEL, EXTENSION,  INT,     "EXTLEVEL", false},
   {TTYPExxx, EXTENSION,  STRING,  "TTYPE",    false},
   {TUNITxxx, EXTENSION,  STRING,  "TUNIT",    false},
   {TNULLxxx, EXTENSION,  STRING,  "TNULL",    false},
   {TSCALxxx, EXTENSION,  INT,     "TSCAL",    false},
   {TZEROxxx, EXTENSION,  INT,     "TZERO",    false},
   {TDISPxxx, EXTENSION,  STRING,  "TDISP",    false},
   {THEAP   , EXTENSION,  INT,     "THEAP   ", false},
   {TDIMxxxx, EXTENSION,  STRING,  "TDIM",     false},

   {END     , MANDATORY,  EMPTY,   "END     ", false} // END here!!!
};

// FitsKnownList constants
const int FitsKnownList::BITPIX_CHAR = 8;
const int FitsKnownList::BITPIX_SHORT = 16;
const int FitsKnownList::BITPIX_INT = 32;
const int FitsKnownList::BITPIX_FLOAT = -32;
const int FitsKnownList::BITPIX_DOUBLE = -64;
const int FitsKnownList::NAXIS_LO_LIMIT = 0;
const int FitsKnownList::NAXIS_HI_LIMIT = 999;

}


