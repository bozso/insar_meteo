/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id$
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

#include <ltl/fitsio.h>

// FitsInBinTable implementation

namespace ltl {

BinTable::BinTable() :
   nrow_(0),
   tfields_( 0 ), tformT_(NULL), tformr_(NULL), tcoloff_(NULL), tforma_( ),
   tscal_(NULL), tzero_(NULL), theap_(0)
{ }

BinTable::BinTable(const int nrow) :
   nrow_(nrow),
   tfields_( 0 ), tformT_(NULL), tformr_(NULL), tcoloff_(NULL), tforma_( ),
   tscal_(NULL), tzero_(NULL), theap_(0)
{ }

BinTable::~BinTable(){
   if(tcoloff_) delete [] tcoloff_;
   if(tformr_) delete [] tformr_;
   if(tformT_) delete [] tformT_;
   tforma_.clear();
   if(tscal_) delete [] tscal_;
   if(tzero_) delete [] tzero_;
}

int BinTable::getTfields(const int i) const
{ return tfields_; }

char BinTable::getTformt(const int i) const
{
   if( (i <= 0) || (i > tfields_) )
      throw FitsException("Request for non-existing column.");
   return tformT_[i-1];
}

size_t BinTable::getTformr(const int i) const
{
   if( (i <= 0) || (i > tfields_) )
      throw FitsException("Request for non-existing column.");
   return tformr_[i-1];
}

FitsBinTableIn::FitsBinTableIn(const FitsIn& other) :
   BinTable(), FitsIn(other)
{
   tfields_ = getInt(string("TFIELDS "));
   tforma_.reserve(tfields_);
   if( getString("XTENSION") != string("BINTABLE") )
      throw FitsException( string("Failed to open ") + getString("XTENSION") +
                           string(" of ") + filename_ +
                           string(" as a BINTABLE extension."));
   // check on mandatory settings
   if(bitpix_ != 8)
      throw FitsException( string("Illegal BITPIX value for BINTABLE.") );
   if(naxis_ != 2)
      throw FitsException( string("Illegal NAXIS value for BINTABLE.") );
   if(gcount_ != 1)
      throw FitsException( string("Illegal GCOUNT value for BINTABLE.") );
   if(tfields_ < KNOWN.NAXIS_LO_LIMIT || tfields_ > KNOWN.NAXIS_HI_LIMIT)
      throw FitsException( string("Illegal TFIELDS value.") );
   // build arrays of column offsets, repeat counts, type id, and check consistency
   tcoloff_ = new off_t[tfields_+1];
   tcoloff_[0] = 0;
   tformr_ = new size_t[tfields_];
   tformT_ = new char[tfields_];
   tforma_.clear();
   char buf[9];
   for(int i = 1; i<=tfields_; ++i){
      sprintf(buf, "TFORM%-3d", i);
      const string tformstr = getString(string(buf));
      const size_t tformtypeoff = tformstr.find_first_of("LXBIJAEDCMP");
      if(tformtypeoff == string::npos)
         throw FitsException( string(" has no legal data type value.") );
      const size_t repeatcount = (tformtypeoff > 0) ? ::strtol(tformstr.c_str() , NULL, 10) : 1;
      tformr_[i-1] = repeatcount;
      tcoloff_[i] = tcoloff_[i-1];
      const char tformtype = tformstr[tformtypeoff];
      tformT_[i-1] = tformtype;
      tcoloff_[i] += ( (tformtype == 'L' || tformtype == 'B' || tformtype == 'A') ? 1 :
                       ( (tformtype == 'I') ? 2 :
                         ( (tformtype == 'J' || tformtype == 'E') ? 4 :
                           ( (tformtype == 'D' || tformtype == 'C' || tformtype == 'P') ? 8 :
                             ( (tformtype == 'M') ? 16 :
                               ( (tformtype == 'X' && repeatcount > 0) ? ( (repeatcount / 8) + 1 ) : 0
                                  ) ) ) ) ) ) * repeatcount;
      tforma_.push_back( (tformstr.size() > tformtypeoff+1) ? tformstr.substr(tformtypeoff+1) : string("") );
   };
   if( tcoloff_[tfields_] != getNaxis(1) )
      throw FitsException( string("TFORMn keys do not sum up to NAXIS1 in BINTABLE extension.") );

   tscal_ = new double[tfields_];
   tzero_ = new double[tfields_];
   FitsCard* the_card = NULL;
   for(int i = 1; i<=tfields_; ++i){
      sprintf(buf, "TSCAL%-3d", i);
      the_card = findCardInList(string(buf), extension_);
      tscal_[i-1] = (the_card == NULL) ? 1.0 : the_card->getFloat();
      buf[1]='Z';buf[2]='E';buf[3]='R';buf[4]='O';//sprintf(buf, "TZERO%-3d", i);
      the_card = findCardInList(string(buf), extension_);
      tzero_[i-1] = (the_card == NULL) ? 0.0 : the_card->getFloat();
   }
   if(pcount_ > 0){
      the_card = findCardInList(string("THEAP   "), extension_);
      theap_ = (the_card == NULL) ? data_length_ : the_card->getInt();
      if(theap_ < data_length_)
         throw FitsException("THEAP value illegal (i.e. points to static table part)");
      if(theap_ >= data_length_ + pcount_)
         throw FitsException("THEAP value illegal (i.e. points beyond data indicated by PCOUNT)");
   }
}

}


