/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_header.cpp 558 2015-03-11 18:17:04Z cag $
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

#include <ltl/io/fits_header.h>

using std::exception;
using std::string;
using std::list;
using std::ostream;
using std::endl;
using std::cerr;

using util::StringFactory;
using util::FitsDate;
using util::Region;
using util::UTDateException;
using util::StringException;

namespace ltl {

// ------------------------------------------------------------------------
// FitsHeader implementations

// public

// constructors
FitsHeader::FitsHeader(unsigned char* inmemptr, size_t inmemlen, bool quiet_please,
                       const bool alljunk) :
   shutup_(quiet_please),
   bitpix_(8), bytpix_(1), naxis_(-1), naxis_array_(NULL),
   extended_(false), pcount_(0), gcount_(1),
   bscale_(1.0), bzero_(0.0),
   data_offset_(off_t(0)), data_length_(off_t(0))
{
   readHeader(string(), alljunk, inmemptr, inmemlen);
}

FitsHeader::FitsHeader(const string& filename, const bool quiet_please,
                       const bool alljunk) :
   shutup_(quiet_please),
   bitpix_(8), bytpix_(1), naxis_(-1), naxis_array_(NULL),
   extended_(false), pcount_(0), gcount_(1),
   bscale_(1.0), bzero_(0.0),
   data_offset_(off_t(0)), data_length_(off_t(0))
{
   readHeader(filename, alljunk);
}

FitsHeader::FitsHeader(const FitsHeader& other, const bool quiet_please,
                       const bool alljunk) :
   naxis_array_(NULL)
{
   copy(other);
   shutup_ = quiet_please;
}

// construct extension (protected)
FitsHeader::FitsHeader(const string& filename, const bool quiet_please,
                       const bool alljunk, const off_t startoffset) :
   shutup_(quiet_please),
   bitpix_(8), bytpix_(1), naxis_(-1), naxis_array_(NULL),
   extended_(true), pcount_(0), gcount_(1),
   bscale_(1.0), bzero_(0.0),
   data_offset_(startoffset), data_length_(off_t(0))
{
   readHeader(filename, alljunk);
}

// empty constructor protected!
FitsHeader::FitsHeader() :
   shutup_(false),
   bitpix_(8), bytpix_(1), naxis_(0), naxis_array_(NULL),
   extended_(false), pcount_(0), gcount_(1),
   bscale_(1.0), bzero_(0.0),
   data_offset_(off_t(PH_C.RECORD_LENGTH)), data_length_(off_t(0))
{ }

FitsHeader::~FitsHeader()
{
   if(naxis_array_)
   {
      delete [] naxis_array_;
      naxis_array_ = NULL;
   }
   clearCardList(mandatory_);
   clearCardList(naxis_i_);
   clearCardList(extension_);
   clearCardList(observe_);
   clearCardList(comment_);
   clearCardList(history_);
   clearCardList(blank_);
   clearCardList(array_);
   clearCardList(otherkeys_);
   clearCardList(junk_);
}

FitsHeader& FitsHeader::operator=(const FitsHeader& other)
{
   if( (&other) != this)
      copy(other);
   return *this;
}

   void FitsHeader::describeSelf( ostream& os )
{
   os << "SIMPLE = T, BITPIX = " << bitpix_ << ", NAXIS = " << naxis_ << endl;
   for( int i = 0; i < naxis_; ++i )
      os << "  NAXIS" << (i+1) << " = " << naxis_array_[i] << endl;
}

// !!! trailing blanks are significant for keywords
// if you want to add missing trailing blank you can use this function
string FitsHeader::adjustKeyword(string keyword) const
{
   keyword.resize(PH_C.KEYWORD_LENGTH, ' ');
   return keyword;
}


// add a pure comment card known reserved or junk
void FitsHeader::addCommentCard(const string& keyword, const string& comment)
{
   if( (keyword == string(KNOWN.KEY[COMMENT].WORD)) ||
       (keyword == string(KNOWN.KEY[HISTORY].WORD)) ||
       (keyword == string(KNOWN.KEY[Blank___].WORD)) )
      addCommentCard(new FitsCommentCard(keyword, comment));
   else
      addJunkCard(keyword, comment);
}

void FitsHeader::addHistory(const string& history)
{
   parseCommentToCardList("HISTORY ", history);
}

void FitsHeader::addComment(const string& comment)
{
   parseCommentToCardList("COMMENT ", comment);
}

void FitsHeader::addValueCard(const string& keyword, const string& value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsStringCard(keyword, value, fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const char* value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsStringCard(keyword, 
                                   string(value), fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const bool value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsBoolCard(keyword, value, fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const long value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsIntCard(keyword, value, fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const int value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsIntCard(keyword, (long)value, fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const float value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsFloatCard(keyword, double(value), fixed, comment));
}

void FitsHeader::addValueCard(const string& keyword, const double value,
                              const string comment , const bool fixed )
{
   addValueCard(new FitsFloatCard(keyword, value, fixed, comment));
}

// erase a card, 0 on success, -1 on keyword not found
// exception on forbidden erasures
int FitsHeader::eraseCard(const string& keyword) throw(FitsException)
{
   // check all lists for keyword
   if( findCardInList(keyword, mandatory_) || findCardInList(keyword, naxis_i_))
      throw FitsException("cannot erase mandatory card '" + keyword + "'");
   bool nothing_erased = false;
   // check if comment or history
   if(keyword == string(KNOWN.KEY[COMMENT].WORD))
      clearCardList(comment_);
   else
   {
      if(keyword == string(KNOWN.KEY[HISTORY].WORD))
         clearCardList(history_);
      else
      {
         if(keyword == string(KNOWN.KEY[Blank___].WORD))
            clearCardList(blank_);
         else
         {
            if(eraseCardFromList(keyword, extension_))
               if(eraseCardFromList(keyword, array_))
                  if(eraseCardFromList(keyword, observe_))
                     if(eraseCardFromList(keyword, otherkeys_))
                     {
                        nothing_erased = true;
                        //throw FitsException("have not found keyword");
                     }
         }
      }
   }
   // also clear junk list of that key
   if(nothing_erased && (eraseCardsFromJunkList(keyword) == 0))
      return -1;
   return 0;
}


// return offset in file to data segement
off_t FitsHeader::getDataOffset() const
{
   return data_offset_;
}

// return mandatorys and array keys
int FitsHeader::getBitpix() const
{
   return bitpix_;
}
int FitsHeader::getBytpix() const
{
   return bytpix_;
}
int FitsHeader::getNaxis() const
{
   return naxis_;
}
int FitsHeader::getNaxis(const int i) const
{
   if( (i <= 0) || (i > naxis_) )
      throw FitsException("Request for unset NAXIS parameter");
   return naxis_array_[i-1];
}
double FitsHeader::getBscale() const
{
   return bscale_;
}
double FitsHeader::getBzero() const
{
   return bzero_;
}
void FitsHeader::setBscale(const double value)
{
   addValueCard("BSCALE  ", value, "REAL = DATA * BSCALE + BZERO");
}
void FitsHeader::setBzero(const double value)
{
   addValueCard("BZERO   ", value, "DATA-OFFSET");
}

off_t FitsHeader::getDataLength() const
{
   return data_length_;
}

Region FitsHeader::getFullRegion() const
{
   Region region(naxis_);
   for(size_t i = 1; i <= region.getDim(); ++i)
      region.setRange(i, 1, naxis_array_[i-1]);
   return region;
}


// return values, exception on type violation
string FitsHeader::getString(const string& keyword) const
{
   return (*(getValueCard(keyword))).getString();
}
bool FitsHeader::getBool(const string& keyword) const
{
   return (*(getValueCard(keyword))).getBool();
}
long FitsHeader::getInt(const string& keyword) const
{
   return (*(getValueCard(keyword))).getInt();
}
double FitsHeader::getFloat(const string& keyword) const
{
   return (*(getValueCard(keyword))).getFloat();
}
bool FitsHeader::isFixed(const string& keyword) const
{
   return (*(getValueCard(keyword))).isFixed();
}

string FitsHeader::getValueAsString(const string& keyword) const
{
   return (*(getValueCard(keyword))).toString();
}


// return comment of a value holding key or line broken content of commentary key
string FitsHeader::getComment(const string& keyword) const
{
   string commentstring;
   if(keyword == string(KNOWN.KEY[COMMENT].WORD))
      commentstring = writeCommentsOfList(comment_);
   else
   {
      if(keyword == string(KNOWN.KEY[HISTORY].WORD))
         commentstring = writeCommentsOfList(history_);
      else
      {
         if(keyword == string(KNOWN.KEY[Blank___].WORD))
            commentstring = writeCommentsOfList(blank_);
         else
            commentstring = (*(getValueCard(keyword))).getComment();
      }
   }
   return commentstring;
}

string FitsHeader::getComment() const
{
   return getComment(string("COMMENT "));
}
string FitsHeader::getHistory() const
{
   return getComment(string("HISTORY "));
}

// build a new valid FITS file format header
string FitsHeader::writeHeader(const bool with_junk)
{
   // reset DATE
   try
   {
      FitsDate now;
      addValueCard("DATE    ", now.toString(), "date and time of writing file");
   }
   catch(util::UTDateException& utde)
   {
      throw FitsException(utde.what());
   }

   // write all lists for keyword
   string header = "";
   header += writeCardsOfList(mandatory_);
   header += writeCardsOfList(naxis_i_);
   header += writeCardsOfList(extension_);
   header += writeCardsOfList(array_);
   header += writeCardsOfList(observe_);
   header += writeCardsOfList(otherkeys_);
   if(with_junk)
      header += writeCardsOfList(junk_);
   header += writeCardsOfList(comment_);
   header += writeCardsOfList(history_);
   header += writeCardsOfList(blank_);
   header += "END";
   header.append(string::size_type(PH_C.RECORD_LENGTH -
                                   ( header.size() % PH_C.RECORD_LENGTH )), ' ');
   return header;
}

// protected

// construct from file
void FitsHeader::readHeader(const string& filename, const bool alljunk, unsigned char* inmemptr, size_t inmemlen)
   throw(FitsException)
{
   bool	inmem = filename.empty();
   int	fd = -1;

   if (!inmem)
   {
      fd = open(filename.c_str(), O_RDONLY);
	  if (fd < 0)
		 throw FitsException("cannot open file '" +
							 filename + "' for reading");

	  struct stat fdstat;
	  fstat(fd, &fdstat);
	  if (data_offset_ >= fdstat.st_size)
	     throw FitsException("no further extensions");
   }
   else
	  if (data_offset_ >= (off_t)inmemlen)
		 throw FitsException("no further extensions");

   const off_t psize = off_t(getpagesize());
   off_t fileoff = -1;
   size_t bytelength = 0;
   char * fitshdrmap = NULL;

   string::size_type not_ready = 0;
   do
   {
      const off_t next_fileoff = (data_offset_ / psize) * psize;
      const size_t next_bytelength = size_t(
         ( ((data_offset_ % psize) + PH_C.RECORD_LENGTH) / psize + 1 ) * psize);
      if( (next_fileoff != fileoff) || (next_bytelength != bytelength) )
      {
         if(fitshdrmap != NULL)
         {
            if (!inmem)
            	munmap(fitshdrmap, bytelength);
            fitshdrmap = NULL;
         }
         fileoff = next_fileoff;
         bytelength = next_bytelength;
         if (!inmem)
         {
			 fitshdrmap = (char *) mmap( NULL, bytelength,
										 PROT_READ, MAP_PRIVATE,
										 fd, fileoff );
			 if(fitshdrmap == (char *) MAP_FAILED)
			 {
				close(fd);
				throw FitsException("cannot map file '" +
									filename + "' for reading");
			 }
         }
         else
         {
        	 if (fileoff > (off_t)inmemlen)
 				throw FitsException("file offset exceeds length of in-memory file");
			 fitshdrmap = (char *)inmemptr + fileoff;
         }

         //cerr << "data offset: " << data_offset_
         //     << ", file offset: " << fileoff << endl;
         //cerr << endl << string(fitshdrmap + 80 - (fileoff%80),2880) << endl;
      }

      char * const record = fitshdrmap +
         size_t(data_offset_ - fileoff);      
      not_ready = parseRecord(string(record, PH_C.RECORD_LENGTH),
                              alljunk);
      data_offset_ += off_t(not_ready);
   } while(not_ready);
   data_offset_ += off_t(PH_C.RECORD_LENGTH);
   
   if(fitshdrmap != NULL)
   {
      if (!inmem)
    	 munmap((char *)fitshdrmap, bytelength);
      fitshdrmap = NULL;
   }
   if (!inmem)
	  close(fd);

   checkNaxis(); // check Naxis keys and set data_length_
}

// parse a single record
string::size_type FitsHeader::parseRecord(const string& record,
                                          const bool alljunk)
   throw(FitsException)
{
   string::size_type i = 0;
   // parse mandatory keywords
   if(naxis_ == -1)
      i = parseMandatory(record);
   else
   {
      if( int(naxis_i_.size()) < naxis_ )
         i = parseNaxis(record);
   }
   
   // parse non mandatory keywords
   while(i < PH_C.RECORD_LENGTH)
   {
      const string cardstring = record.substr(i, PH_C.CARD_LENGTH);
      // check on "END     "
      if(cardstring.substr(string::size_type(0), PH_C.KEYWORD_LENGTH) ==
         string(KNOWN.KEY[END].WORD))
      {
         if(!alljunk)
         {
            const string::size_type last_unblank = record.find_last_not_of(' ');
            if( (last_unblank != (i + 2)) )
            {
               try
               {
                  const string exceptiontext =
                     "Header corrupted after END\nEND: " +
                     StringFactory::toString(int(i), 4) + ", non blank: " +
                     StringFactory::toString(int(record.find_last_not_of(' ')), 4);
                  throw FitsException(exceptiontext);
               }
               catch(util::StringException& stre)
               {
                  const string excepexceptiontext =
                     string("Header corrupted after END\nand caught exception while parsing exception: ") +
                     string(stre.what());
                  throw FitsException(excepexceptiontext);
               }
            }
         }
         return string::size_type(0);
      }
      try
      {
         // dont parse keys, just put them on list
         if(alljunk)
            addJunkCard(cardstring);
         else
         { // parse the keys
            FitsCard* nextone = parseCard(cardstring);
            if(nextone != NULL)
            {
               if((*nextone).isJunk())
                  addJunkCard(nextone);
               else
               {
                  if((*nextone).isCommentary())
                     addCommentCard(nextone);
                  else
                     addValueCard(nextone);
               }
            }
         }
      }
      catch (FitsException& e)
      {
         if(!shutup_)
            cerr << cardstring << endl
                 << e.what()
                 << "\nAdding to junk list\n";

         addJunkCard(cardstring);
      }
      i += PH_C.CARD_LENGTH;
   }
   return i;
}

// parse mandatory keywords
// return No parsed bytes
string::size_type FitsHeader::parseMandatory(const string& record)
   throw(FitsException)
{
   string::size_type i = 0;
   // read SIMPLE, BITPIX and NAXIS
   while(i < (3 * PH_C.CARD_LENGTH))
   {
      const string cardstring = record.substr(i, PH_C.CARD_LENGTH);
      mandatory_.push_back( parseCard(cardstring) );
      i += PH_C.CARD_LENGTH;
   }
   // check SIMPLE / XTENSION, BITPIX and NAXIS
   MandatoryIter j = mandatory_.begin();
   if(data_offset_ == 0) // primary header
   {
      if((**j).what_id() != SIMPLE)
      {
         clearCardList(mandatory_);
         throw FitsException("FITS file not SIMPLE");
      }
      if( ! (**j).getBool() )
      {
         clearCardList(mandatory_);
         throw FitsException("FITS file not SIMPLE (SIMPLE not true)");
      }
   }
   else // extension header
   {
      if((**j).what_id() != XTENSION)
      {
         clearCardList(mandatory_);
         throw FitsException("FITS extension does not start with XTENSION");
      }
//       if( ((**j).getString() != string("IMAGE")) )
//       {
//          clearCardList(mandatory_);
//          throw FitsException("Sorry only IMAGE Extension is supported 'til now.");
//       }
   }
   ++j;
   if( (**j).what_id() != BITPIX )
   {
      clearCardList(mandatory_);
      throw FitsException("FITS file has no BITPIX");
   }
   // init bitpix
   bitpix_ = (**j).getInt(); //cerr << "bitpix: " << bitpix_ << endl;
   if( (bitpix_ != KNOWN.BITPIX_CHAR) && (bitpix_ != KNOWN.BITPIX_SHORT) &&
       (bitpix_ != KNOWN.BITPIX_INT) && (bitpix_ != KNOWN.BITPIX_FLOAT) &&
       (bitpix_ != KNOWN.BITPIX_DOUBLE) )
   {
      clearCardList(mandatory_);
      throw FitsException("FITS file has invalid BITPIX (not 8, 16, 32, -32 , -64)");
   }
   bytpix_ = abs(bitpix_) / 8;
   ++j;
   if((**j).getKeyword() != (string(KNOWN.KEY[NAXISxxx].WORD) + "   "))
   {
      clearCardList(mandatory_);
      throw FitsException("FITS file has no NAXIS");
   }
   // init naxis
   naxis_ = (**j).getInt(); //cerr<<"naxis: "<<naxis_<<endl;
   if( (naxis_ < KNOWN.NAXIS_LO_LIMIT) || (naxis_ > KNOWN.NAXIS_HI_LIMIT) )
   {
      clearCardList(mandatory_);
      throw FitsException("FITS file has invalid NAXIS ( < 0 or > 999)");
   }
   // continue with NAXIS list
   if(naxis_)
      i = parseNaxis(record, i);
   return i;
}

// parse naxis list
// return No already parsed bytes of record
string::size_type FitsHeader::parseNaxis(const string& record,
                                         string::size_type card_of_record )
{
   // read the naxis list
   while( (int(naxis_i_.size()) < naxis_) &&
          (card_of_record < PH_C.RECORD_LENGTH) )
   {
      const string cardstring = record.substr(card_of_record, PH_C.CARD_LENGTH);
      naxis_i_.push_back( parseCard(cardstring) );
      card_of_record += PH_C.CARD_LENGTH;
   }
   return card_of_record;
}

// check syntax of naxis list and fill short cut array
void FitsHeader::checkNaxis() throw(FitsException)
{
   if(naxis_array_)
   {
      delete [] naxis_array_;
      naxis_array_ = NULL;
   }
   if(naxis_ > 0)
   {
      naxis_array_ = new int [size_t(naxis_)];
      int k = 1;
      data_length_ = off_t(1);
      for(list<FitsCard *>::iterator j = naxis_i_.begin();
          j != naxis_i_.end(); ++j)
      {
         //cerr << "naxis" << k << ": " << (**j).getInt() << endl;
         const string key = (**j).getKeyword();
         if( (**j).what_id() != NAXISxxx)
         {
            clearCardList(naxis_i_);
            throw FitsException("invalid NAXIS setting (NAXIS key missing)");
         }
         if(atoi( (key.substr(5,3)).c_str() ) != k)
         {
            clearCardList(naxis_i_);
            throw FitsException("invalid NAXIS list (NAXIS keys missordered)");
         }
         const int valueholder = (**j).getInt();
         if( valueholder <= 0)
         {
            clearCardList(naxis_i_);
            throw FitsException("invalid or insane NAXIS setting (NAXIS negative or 0)");
         }
         naxis_array_[k-1] = valueholder;
         data_length_ *= off_t(valueholder);
         ++k;
      }
      data_length_ += off_t(pcount_);
      data_length_ *= off_t(bytpix_ * gcount_);
   }
   else
      data_length_ = off_t(0);
}

// parse a complete card
// (complex is still missing)
FitsCard * FitsHeader::parseCard(const string& card) const
   throw(FitsException)
{
   const string keyword = card.substr(string::size_type(0), PH_C.KEYWORD_LENGTH);
   // syntax check
   const string::size_type keywordstart = keyword.find_first_not_of(' ');
   // keyword is blank? goto comment type, otherwise stay here and continue parse
   if(keywordstart != keyword.npos)
   {
      // value indicator?
      if(card.substr(PH_C.KEYWORD_LENGTH, 2) == "= ")
      {
         bool fixed = false;
         // check on type
         string::size_type valuestart =
            card.find_first_not_of(' ', string::size_type(10));
         if(valuestart == card.npos)
            throw FitsException("value is undefined");
         // string ?
         if(card[valuestart] == '\'')
         {
            ++valuestart; // first char of string
            string::size_type valueend = valuestart;
            while( true )
            { // this is not beautiful but necessary
               valueend = card.find_first_of('\'', valueend);
               if(valueend == card.npos)
                  throw FitsException("closing quote of string is missing");
               if( valueend == PH_C.CARD_LENGTH - 1 )
                  break; // closing quote is last char
               // next test must be done AFTER above one,
               // if above is true valueend + 1 would violate string range
               if( card[valueend+1] != '\'' )
                  break; // it is a closing quote, no ''
               valueend += 2;
               if( valueend >= PH_C.CARD_LENGTH )
                  throw FitsException("quote missmatch in string");
            }
            // now valueend points to closing quote
            fixed = false;
            string valuestring = "";
            if(valueend != valuestart) // not empty string?
            {
               // fixed format?
               if( (valuestart == (PH_C.VALUE_MIN_START + 1)) &&
                   (valueend >= PH_C.FIXED_STRING_MIN_END) )
                  fixed = true;
               // blank string or not?
               const string::size_type stringend =
                  card.find_last_not_of(' ', valueend - 1);
               valuestring = (stringend < valuestart) ?
                  string(" ") :
                  card.substr(valuestart, stringend - valuestart + 1);
            }
            //cerr << keyword << " is a string: " << valuestring << endl;
            return new FitsStringCard(
               keyword, valuestring, fixed,
               getCardComment(card.substr(valueend + 1,
                                          PH_C.CARD_LENGTH - valueend)),
               shutup_ ? 0 : 1);
         }
         else
         { // no string
            // bool ?
            if( (card[valuestart] == 'T') || (card[valuestart] == 'F') )
            {
               bool value = false;
               if(card[valuestart] == 'T')
                  value = true;
               bool fixed = false;
               if(valuestart == PH_C.FIXED_VALUE_END - 1)
                  fixed = true;
               //cerr << keyword << " is a bool: " << value << endl;
               return new FitsBoolCard(
                  keyword, value, fixed,
                  getCardComment(card.substr(valuestart + 1,
                                             PH_C.CARD_LENGTH - valuestart)) );
            }
            else // number or invalid
            {
               char *intendptr;
               char *floatendptr;
//                const char *valuestartptr =
//                   (card.substr(PH_C.VALUE_MIN_START,
//                                PH_C.CARD_LENGTH - PH_C.VALUE_MIN_START)).c_str();
               const string valuestring = card.substr(PH_C.VALUE_MIN_START,
                                                      PH_C.CARD_LENGTH - PH_C.VALUE_MIN_START);
               const char *valuestartptr = valuestring.c_str();
               const long intvalue = strtol(valuestartptr , &intendptr, 10);
               const double floatvalue = strtod(valuestartptr, &floatendptr);
               if(floatendptr != valuestartptr) // it is a number
               {
                  // is it fixed?
                  bool fixed = false;
                  const string::size_type valuelength =
                     string::size_type(floatendptr - valuestartptr);
                  if( valuelength == (PH_C.FIXED_VALUE_END -
                                      PH_C.VALUE_MIN_START) )
                     fixed = true;
                  const string::size_type valueend =
                     PH_C.VALUE_MIN_START + valuelength;
                  if(intendptr == floatendptr) // it is a integer
                  {
                     //cerr << keyword << " is an int: " << intvalue << endl;
                     return new FitsIntCard(
                        keyword, intvalue, fixed,
                        getCardComment( card.substr(valueend,
                                                    PH_C.CARD_LENGTH - valueend)
                           ) );
                  }
                  // it is a float
                  //cerr << keyword << " is a float: " << floatvalue << endl;
                  return new FitsFloatCard(
                     keyword, floatvalue, fixed,
                     getCardComment( card.substr(valueend,
                                                 PH_C.CARD_LENGTH - valueend)
                        ) );
               }
               else // comment ?
               {
                  if(card.find_first_of('/', PH_C.VALUE_MIN_START) == card.npos)
                  {
                     const string exceptiontext = "neither value nor comment";
                     throw FitsException(exceptiontext);
                  }
                  // go to junk
               }
            }
         }
      } // value card end
      // no value indicator
      else
      {
         // is it a known comment?
         if( (keyword == string(KNOWN.KEY[COMMENT].WORD)) ||
             (keyword == string(KNOWN.KEY[HISTORY].WORD)) )
         {
            string::size_type commentstart =
               card.find_first_not_of(' ', PH_C.KEYWORD_LENGTH);
            if(commentstart == card.npos)
               return NULL; // empty
            const string commentstring =
               card.substr(commentstart,PH_C.CARD_LENGTH - commentstart);
            //cerr << keyword << " is a comment: " << commentstring << endl;
            return new FitsCommentCard(keyword, commentstring);
         }
      }
   }
   // blank keyword ?
   if(keyword == string(KNOWN.KEY[Blank___].WORD))
   {
      string::size_type commentstart =
         card.find_first_not_of(' ', PH_C.KEYWORD_LENGTH);
      if(commentstart == card.npos)
         return NULL; // empty
      const string commentstring =
         card.substr(PH_C.KEYWORD_LENGTH,
                     PH_C.CARD_LENGTH - PH_C.KEYWORD_LENGTH);
      //cerr << keyword << " is a blank: " << commentstring << endl;
      return new FitsCommentCard(keyword, commentstring);
   }
   // ... seems to be just junk
   //cerr << keyword << " is just junk: " << card << endl;
   return new FitsJunkCard(
      keyword,
      card.substr(PH_C.KEYWORD_LENGTH,
                  PH_C.CARD_LENGTH - PH_C.KEYWORD_LENGTH));
}

// check in teststring if a comment follows after the value
string FitsHeader::getCardComment(const string& teststring) const
   throw(FitsException)
{
   // check on comment
   const string::size_type commentstart = teststring.find_first_not_of(' ');
   if(commentstart != teststring.npos)
   {
      if(teststring[commentstart] != '/')
         throw FitsException("neither comment nore blanks after value");
      const string::size_type commenttext =
         teststring.find_first_not_of(' ', commentstart + 1);
      if(commenttext != teststring.npos)
      {
         const string::size_type commentend = teststring.find_last_not_of(' ');
         if(commentend >= commenttext)
            return teststring.substr(commenttext, commentend - commenttext + 1);
      }
   }
   return string("");
}

// parse and add comment
void FitsHeader::parseCommentToCardList(const string& keyword,
                                        const string& comment)
{
   string::size_type linestart = string::size_type(0);
   // offset to last possible char
   string::size_type horizont = linestart + PH_C.COMMENT_MAX_LENGTH;
   while( comment.size() > horizont )
   {
      string::size_type lineend =
         comment.find_last_of(' ', horizont); // find space, backwards
      if(lineend != comment.npos)
         lineend = comment.find_last_not_of(' ', lineend); // find last char
      if((lineend < linestart) || (lineend == comment.npos))
         lineend = horizont; // looooong word?
      const string line = comment.substr(linestart, lineend - linestart + 1);
      addCommentCard(keyword, line);
      linestart = lineend + 1;
      horizont = linestart + PH_C.COMMENT_MAX_LENGTH - 1;
   }
   const string line = comment.substr(linestart, comment.size() - linestart);
   addCommentCard(keyword, line);
}

// add a commentary card to the correct list
void FitsHeader::addCommentCard(FitsCard* cardptr)
   throw(FitsException)
{
   if( (*cardptr).isCommentary() )
   {
      switch( (*cardptr).what_id() )
      {
         case COMMENT:
            comment_.push_back(cardptr);
            break;
         case HISTORY:
            history_.push_back(cardptr);
            break;
         case Blank___:
            blank_.push_back(cardptr);
            break;
         default:
            delete cardptr;
            throw FitsException(
               "strange! got commentary but not COMMENT, HISTORY or blank");
            break;
      }
   }
   else
   {
      delete cardptr;
      throw FitsException("no commentary card, try addValueCard()");
   }
}

// just split 80 char string into keyword + rest and push it on the
// junk list
void FitsHeader::addJunkCard(const string& native_card)
{
   addJunkCard( native_card.substr(string::size_type(0), PH_C.KEYWORD_LENGTH),
                native_card.substr(PH_C.KEYWORD_LENGTH,
                                   PH_C.CARD_LENGTH - PH_C.KEYWORD_LENGTH) );
}
// add a suitable trimmed junk card
void FitsHeader::addJunkCard(const string& keyword, const string& comment)
{
   addJunkCard(new FitsJunkCard(keyword, comment));
}
// add a already existing FitsCard to junk list
void FitsHeader::addJunkCard(FitsCard* cardptr)
{
   junk_.push_back(cardptr);
}

// put a new card on the proper list
void FitsHeader::addValueCard(FitsCard* cardptr) throw(FitsException)
{

   // primitive type and syntax check is done with card construction
   // now check sophisticated cases

   // first handle reserved keys cases
   if( (*cardptr).isReserved() )
   {

      // don't allow mandatorys to be set here
      if( (*cardptr).isMandatory() )
      {
         //delete cardptr;
         throw FitsException(
            "mandatory keywords may only be (re)set via constructor");
      }
      // commentary card?
      if( (*cardptr).isCommentary() )
      {
         //delete cardptr;
         throw FitsException("commentary card, try addCommentCard()");
      }
      // extensions: only common extension mandatories are checked for here
      if( (*cardptr).isExtension() )
      {
         eraseCardsFromJunkList( (*cardptr).getKeyword() );
         eraseCardFromList( (*cardptr).getKeyword(), extension_ );
         switch( (*cardptr).what_id() )
         {
            case EXTEND:
               extended_ = (*cardptr).getBool();
               break;// set value class intern
            case PCOUNT:
               pcount_ = (*cardptr).getInt();
               break;// set value class intern
            case GCOUNT:
               gcount_ = (*cardptr).getInt();
               break;// set value class intern
            default:
               break;
         }
         if( ((*cardptr).what_id() == EXTEND) && (data_offset_ == 0) )
            extension_.push_front(cardptr);
         else
            extension_.push_back(cardptr);
         //if(!shutup_)
         //cerr << "Extension keywords not supported, discarding card "
         //<< (*cardptr).getKeyword() << endl;
         //delete cardptr;
         //throw FitsException("extension keywords not supported");
      }
      else
      {
         if( (*cardptr).isArray() )
         {
            switch( (*cardptr).what_id() )
            {
               case BSCALE:
                  bscale_ = (*cardptr).getFloat();
                  break;// set value class intern
               case BZERO:
                  bzero_ = (*cardptr).getFloat();
                  break;// set value class intern
               default:
                  break;
            }
            eraseCardsFromJunkList( (*cardptr).getKeyword() );
            eraseCardFromList( (*cardptr).getKeyword(), array_ );
            array_.push_back(cardptr);
         }
         else // must be of observe type (isDate or isOther)
         {
            eraseCardsFromJunkList( (*cardptr).getKeyword() );
            eraseCardFromList( (*cardptr).getKeyword(), observe_);
            if( (*cardptr).isDate() )
               observe_.push_front(cardptr);
            else
            {
               switch( (*cardptr).what_id() )
               {
                  case BLOCKED:// ignore BLOCKED
                     if(!shutup_)
                        cerr << "BLOCKED is deprecated, discarding card when writing\n";
                     //delete cardptr;
                     break;
                  case EPOCH: // if EQUINOX not set write EPOCH to EQUINOX, otherwise ignore
                     if( findCardInList("EQUINOX ", observe_) == NULL )
                     {
                        const double equinox = (*cardptr).getFloat();
                        FitsFloatCard* equinoxptr =
                           new FitsFloatCard(
                              "EQUINOX ", equinox, true,
                              "taken from deprecated EPOCH keyword");
                        observe_.push_back(equinoxptr);
                        if(!shutup_)
                           cerr << "EPOCH is deprecated, renaming to EQUINOX when writing\n";
                     }
                     else
                        if(!shutup_)
                           cerr << "EPOCH is deprecated, discarding card when writing\n";
                     //delete cardptr;
                     break;
                  default:
                     //observe_.push_back(cardptr);
                     break;
               }
               observe_.push_back(cardptr);
            }
         }
      }
   }
   else // no reserved key
   {
      // must either be junk or simply an other keyword
      if( (*cardptr).isJunk() )
         addJunkCard(cardptr);
      else
      {
         eraseCardsFromJunkList( (*cardptr).getKeyword() );
         eraseCardFromList( (*cardptr).getKeyword(), otherkeys_ );
         otherkeys_.push_back(cardptr);
      }
   }
}

// clear a whole FitsCard list
void FitsHeader::clearCardList(list< FitsCard * > & the_list)
{
   const list< FitsCard * >::iterator list_end = the_list.end();
   for(list< FitsCard * >::iterator iter = the_list.begin();
       iter != list_end;
       ++iter)
      delete *iter;
   the_list.clear();
}

// erase a card, 0 on success, -1 on keyword not found
int FitsHeader::eraseCardFromList(const string& keyword,
                                  list< FitsCard * > & the_list)
{
   const list< FitsCard * >::iterator list_end = the_list.end();
   for(list< FitsCard * >::iterator iter = the_list.begin();
       iter != list_end;
       ++iter)
   {
      if( (**iter).getKeyword() == keyword )
      {
         delete *iter;
         the_list.erase(iter);
         return 0;
      }
   }
   return -1;
}

// erase cards with name keyword from the JunkList,
// returns number of cards erased
int FitsHeader::eraseCardsFromJunkList(const string& keyword)
{
   int on_list = 0;
   JunkIter iter = junk_.begin();
   const JunkIter junk_end = junk_.end();

   while(iter != junk_end)
   {
      if( (**iter).getKeyword() == keyword )
      {
         delete *iter;
         iter = junk_.erase(iter);
         ++on_list;
      }
      else
         ++iter;
   }
   return on_list;
}

// return pointer to card or NULL if not found
FitsCard* FitsHeader::findCardInList(
   const string& keyword,
   const list< FitsCard * > & the_list) const
{
   const list<FitsCard *>::const_iterator list_end = the_list.end();
   for(list<FitsCard *>::const_iterator iter = the_list.begin();
       iter != list_end; ++iter)
      if( (**iter).getKeyword() == keyword )
         return *iter;
   return NULL;
}

// find card in value lists (or junk list)
FitsCard* FitsHeader::getValueCard(const string& keyword) const
   throw(FitsException)
{
   // check all lists for keyword
   FitsCard *the_card;
   if( (the_card = findCardInList(keyword, mandatory_)) == NULL)
      if( (the_card = findCardInList(keyword, naxis_i_)) == NULL)
         if( (the_card = findCardInList(keyword, extension_)) == NULL)
            if( (the_card = findCardInList(keyword, array_)) == NULL)
               if( (the_card = findCardInList(keyword, observe_)) == NULL)
                  if( (the_card = findCardInList(keyword, otherkeys_)) == NULL)
                     if( (the_card = findCardInList(keyword, junk_)) == NULL)
                     {
                        // check if comment or history
                        if( (keyword == string(KNOWN.KEY[COMMENT].WORD)) ||
                            (keyword == string(KNOWN.KEY[HISTORY].WORD)) ||
                            (keyword == string(KNOWN.KEY[Blank___].WORD)) )
                           throw FitsException(
                              "use getComment or getHistory to read keyword '" +
                              keyword + "'" );
                        throw FitsException("cannot find keyword '" +
                                            keyword + "'");
                  }
   return the_card;
}

// returns string holding new line broken comments of a whole list
string FitsHeader::writeCommentsOfList(const list< FitsCard * > & the_list) const
{
   string liststring = "";
   const list<FitsCard *>::const_iterator list_end = the_list.end();
   for(list<FitsCard *>::const_iterator iter = the_list.begin();
       iter != list_end; ++iter)
      liststring += (**iter).getComment()+"\n";
   return liststring;
}

// returns string holding FITS formatted cards of a list
string FitsHeader::writeCardsOfList(const list< FitsCard * > & the_list) const
{
   string liststring = "";
   const list<FitsCard *>::const_iterator list_end = the_list.end();
   for(list<FitsCard *>::const_iterator iter = the_list.begin();
       iter != list_end; ++iter)
      liststring += (**iter).writeCard();
   return liststring;
}

// for copy constructor
void FitsHeader::copy(const FitsHeader& other)
{
   shutup_ = other.shutup_;
   bitpix_ = other.bitpix_;
   bytpix_ = other.bytpix_;
   naxis_ = other.naxis_;
   extended_ = other.extended_;
   pcount_ = other.pcount_;
   gcount_ = other.gcount_;
   bscale_ = other.bscale_;
   bzero_ = other.bzero_;
   data_offset_ = other.data_offset_;
   data_length_ = other.data_length_;

   if(naxis_array_ != NULL)
      delete naxis_array_;
   naxis_array_ = new int[size_t(naxis_)];

   for(int i = 0; i < naxis_; ++i)
      naxis_array_[i] = other.naxis_array_[i];
   copyCardList(mandatory_, other.mandatory_);
   copyCardList(naxis_i_, other.naxis_i_);
   copyCardList(extension_, other.extension_);
   copyCardList(observe_, other.observe_);
   copyCardList(comment_, other.comment_);
   copyCardList(history_, other.history_);
   copyCardList(blank_, other.blank_);
   copyCardList(array_, other.array_);
   copyCardList(otherkeys_, other.otherkeys_);
   copyCardList(junk_, other.junk_);
}

void FitsHeader::copyCardList(list<FitsCard *> & dest_list,
                              const list<FitsCard *> & src_list)
{
   clearCardList(dest_list);
   const list<FitsCard *>::const_iterator src_end = src_list.end();
   for(list<FitsCard *>::const_iterator iter = src_list.begin();
       iter != src_end; ++iter)
      dest_list.push_back( (**iter).clone() );
}

// FitsHeader protected method
void FitsHeader::testRegion(const Region &testreg) const
   throw(FitsException)
{
   // check on valid cutregion
   const Region maxreg = getFullRegion();
   const string illegal_region = string("request for illegal region:\n") +
      testreg.toString() + string(" is no subarray of ") +
      maxreg.toString();
   if(testreg.getDim() != maxreg.getDim())
      throw FitsException(illegal_region);
   for(size_t i = 1; i <= maxreg.getDim(); ++i)
   {
      if(testreg.getStart(i) < maxreg.getStart(i))
         throw FitsException(illegal_region);
      if(testreg.getEnd(i) > maxreg.getEnd(i))
         throw FitsException(illegal_region);
   }
}

}
