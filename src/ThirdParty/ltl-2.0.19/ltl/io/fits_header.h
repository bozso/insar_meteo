/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_header.h 533 2014-02-18 16:28:32Z drory $
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


/*! \file fits_header.h

Holds FITS header class ltl::FitsHeader and methods.

FITS Syntax

Chars 1 - 8 can be:
\li valid (or blank) keyword,
\li invalid (or empty) keyword

Valid keywords:
mandatory, other reserved, or additional

Mandatory keywords:
principal, or conforming extensions.

Principal: all must be fixed format.
\li "SIMPLE  ": must be 1st in header, followed by
\li "BITPIX  ": 8, 16, 32, -32, -64, followed by
\li "NAXIS   ": 0 <= n <1000, integer
\li if NAXIS > 0
\li "NAXIS1  ", ..., "NAXISnnn"
\li ...
\li "END     " and empty after

Conforming extensions:
Support development is on the way...

Other reserved keywords:
\li "DATE    ": 'YYYY-MM-DD[Thh:mm:ss[.sss]]' or 'DD/MM/YY' before 2000
\li "ORIGIN  ": character string
\li "BLOCKED ": will be discarded
\li "DATE-OBS", "DATExxxx": same format as "DATE    "
\li "TELESCOP", "INSTRUME", "OBSERVER", "OBJECT  ": character strings
\li "EQUINOX ": floating point equinox in years
\li "EPOCH   ": will be transformed to "EQUINOX " if "EQUINOX " not present
\li "AUTHOR  ", "REFERENC": character string
\li array keywords ... BSCALE, BZERO, ...

Intensive syntax check is performed only for mandatory keywords
and some reserved keywords.

Values: no complex values supported up to now. \n
Supported: strings, boolian, integers, floating point;\n
all in fixed or unfixed format.

*/

#ifndef __FITS_HEADER_H__
#define __FITS_HEADER_H__

#define FITS_SYNTAX_CHECK

#include <ltl/config.h>

// constants
#include <ltl/io/fits_const.h>

// cards
#include <ltl/io/fits_card.h>

// former fits_util, now directly from orig. util
#include <ltl/util/stringfac.h>
#include <ltl/util/utdate.h>
#include <ltl/util/region.h>

#include <ltl/misc/exceptions.h>

// includes
#include <algorithm>
#include <complex>
//#include <fstream>
#include <iostream>
#include <string> // string

#include <list>
//#include <stl.h>

#include <cmath>
#include <cstdlib> // abs

// mmap file
#include <fcntl.h>
#include <unistd.h>
//#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

// classes

namespace ltl {

//! Representation of a complete FITS header. Holds also header I/O methods.
class FitsHeader
{
   protected:
      //! Boolean indicating if no error messages shall be written to stderr.
      bool shutup_;
      //! Shortcuts to FITS keywords BITPIX, abs(BITPIX)/8 and NAXIS.
      int bitpix_, bytpix_, naxis_;
      //! Pointer to array holding shortcuts to NAXIS keywords.
      int* naxis_array_;
      //! Shortcut indicating extension maybe present.
      bool extended_;
      //! Shortcuts for calculating length of extension.
      int pcount_, gcount_;

      //! Shortcuts to FITS keywords BSCALE and BZERO.
      double bscale_, bzero_;
      //! Offset to first element of data segment,
      off_t data_offset_;
      //! Length of data segment in bytes.
      off_t data_length_;

      //! Structure holding physical FITS sizes.
      static const FitsPhysical PH_C;

      //! List of all known keywords and their types.
      static const FitsKnownList KNOWN;

      /*! \name Card Lists
        Lists representing the FITS header structure.
      */
      //@{
      // mandatory principal
      typedef std::list<FitsCard *> MandatoryList;
      typedef MandatoryList::iterator MandatoryIter;
      //! List holding all mandatory cards.
      MandatoryList mandatory_;
      typedef std::list<FitsCard *> NaxisList;
      typedef NaxisList::iterator NaxisIter;
      //! List holding all NAXIS# cards.
      NaxisList naxis_i_;
      typedef std::list<FitsCard *> ExtensionList;
      typedef ExtensionList::iterator ExtensionIter;
      //! List holding EXTENSION cards.
      ExtensionList extension_;

      // reserved keywords which may be of interest
      // DATE, ORIGIN, BLOCKED will be discarded / reset
      // DATE-OBS, DATExxxx, TELESCOP, INSTRUME, OBSERVER, OBJECT,
      // EQUINOX, EPOCH will be transformed to EQUINOX if EQUINOX not present
      // AUTHOR, REFERENC
      typedef std::list<FitsCard *> ObserveList;
      typedef ObserveList::iterator ObserveIter;
      //! List holding reserved keywords cards which are not on other lists.
      ObserveList observe_;

      // COMMENT
      typedef std::list<FitsCard *> CommentList;
      typedef CommentList::iterator CommentIter;
      //! List holding the COMMENT cards.
      CommentList comment_;

      // HISTORY
      typedef std::list<FitsCard *> HistoryList;
      typedef HistoryList::iterator HistoryIter;
      //! List holding the HISTORY cards
      HistoryList history_;

      // blank keywords holding values
      typedef std::list<FitsCard *> BlankList;
      typedef BlankList::iterator BlankIter;
      //! List holding the blank cards.
      BlankList blank_;

      // BSCALE, BZERO, BUNIT, BLANK, CTYPEn, CRPIXn, CRVALn, CRDELTn, CROTAn,
      // DATAMIN, DATAMAX
      typedef std::list<FitsCard *> ArrayList;
      typedef ArrayList::iterator ArrayIter;
      //! List holding array keywords cards.
      ArrayList array_;

      // other keywords holding values
      typedef std::list<FitsCard *> OtherList;
      typedef OtherList::iterator OtherIter;
      //! List holding all other cards.
      OtherList otherkeys_;

      // junk keywords
      typedef std::list<FitsCard *> JunkList;
      typedef JunkList::iterator JunkIter;
      //! List holding the cards not complying with FITS standard.
      JunkList junk_;
      //@}

   public:
      //! Construct from existing file.
      FitsHeader(unsigned char* inmemptr, size_t inmemlen, const bool quiet_please = false,
                 const bool alljunk = false);
      FitsHeader(const std::string& filename, const bool quiet_please = false,
                 const bool alljunk = false);
      //! Construct as copy from \e other.
      FitsHeader(const FitsHeader& other,
                 const bool quiet_please = false,
                 const bool alljunk = false);
      //! Destruct Object.
      virtual ~FitsHeader();

      //! Copy values from \e other.
      FitsHeader& operator=(const FitsHeader& other);

      //! Write Mandatories to \e os.
      virtual void describeSelf( std::ostream& os );

      //! Return \e keyword trimmed (or expanded) to a width of 8 chars.
      std::string adjustKeyword(std::string keyword) const;

      /*! \name Adding new Cards
      */
      //@{
      //! Add a commentary card.
      void addCommentCard(const std::string& keyword, const std::string& comment);
      //! Add a history of arbitrary length.
      void addHistory(const std::string& history);
      //! Add a history of arbitrary length.
      void addComment(const std::string& comment);
      //! Add a string value card.
      void addValueCard(const std::string& keyword, const std::string& value,
                        const std::string comment = "", const bool fixed = true);
      //! Add a char value card.
      void addValueCard(const std::string& keyword, const char* value,
                        const std::string comment = "", const bool fixed = true);
      //! Add a boolean value card.
      void addValueCard(const std::string& keyword, const bool value,
                        const std::string comment = "", const bool fixed = true);
      //! Add an integer value card.
      void addValueCard(const std::string& keyword, const int value,
                        const std::string comment = "", const bool fixed = true);
      //! Add an integer value card.
      void addValueCard(const std::string& keyword, const long value,
                        const std::string comment = "", const bool fixed = true);
      //! Add a floating point value card.
      void addValueCard(const std::string& keyword, const float value,
                        const std::string comment = "", const bool fixed = true);
      //! Add a floating point value card.
      void addValueCard(const std::string& keyword, const double value,
                        const std::string comment = "", const bool fixed = true);
      //@}
      //! Erase FITS card \e keyword.
      int eraseCard(const std::string& keyword) throw(FitsException);

      //! Return the byte offset within the file to the data segment.
      off_t getDataOffset() const;

      /*! \name Shortcuts
        Get (and set) values of mandatories and array keys via shortcuts.
      */
      //@{
      //! Return BITPIX setting.
      int getBitpix() const;
      //! Return bytes per pixel, i.e. abs( bitpix_ ) / 8.
      int getBytpix() const;
      //! Return NAXIS setting.
      int getNaxis() const;
      //! Return width of \e i NAXIS.
      int getNaxis(const int i) const;
      //! Return BSCALE setting.
      double getBscale() const;
      //! Return BZERO setting.
      double getBzero() const;
      //! Set the BSCALE key to \e value.
      void setBscale(const double value);
      //! Set the BZERO key to \e value.
      void setBzero(const double value);
      //! Return a util::Region according to the NAXIS geometry.
      util::Region getFullRegion() const;
      //@}
      //! Return the size of the data segment in bytes.
      off_t getDataLength() const;

      /*! \name Read Cards
        Search the lists for card matching \e keyword and
        return requested instance.
      */
      //@{
      //! Return string value of FITS key \e keyword.
      std::string getString(const std::string& keyword) const;
      //! Return boolean value of FITS key \e keyword.
      bool getBool(const std::string& keyword) const;
      //! Return integer value of FITS key \e keyword.
      long getInt(const std::string& keyword) const;
      //! Return floating point value of FITS key \e keyword.
      double getFloat(const std::string& keyword) const;
      //! Indicate if value of FITS key \e keyword is of fixed type.
      bool isFixed(const std::string& keyword) const;
      //! Return comment of FITS key \e keyword.
      std::string getComment(const std::string& keyword) const;
      //! Return complete COMMENT.
      std::string getComment() const;
      //! Return complete HISTORY.
      std::string getHistory() const;

      //! Return value of FITS key \e keyword as a string irrespective of its type.
      std::string getValueAsString(const std::string& keyword) const;
      //@}

      //! Return a new valid FITS file format header.
      std::string writeHeader(const bool with_junk = false);

   protected:
      // helpmethods called from public methods

      //! Empty constructor is protected!
      FitsHeader();

      //! Construct extension from existing file.
      FitsHeader(const std::string& filename, const bool quiet_please,
                 const bool alljunk, const off_t startoffset);

      //! Read and parse a FITS header from file.
      void readHeader(const std::string& filename, const bool alljunk = false, unsigned char* inmemptr = NULL, size_t inmemlen = 0)
         throw(FitsException);

      /*! \name Parse Record string
        The next 4 methods parse a complete 2880 char record
        into lists of ltl::FitsCards.
      */
      //@{
      //! Parse a FITS record and assign the cards to their proper lists.
      std::string::size_type parseRecord(const std::string& record,
                                    const bool alljunk = false)
         throw(FitsException);
      //! Parse the mandatory keys of \e record.
      std::string::size_type parseMandatory(const std::string& record)
         throw(FitsException);
      //! Parse cards until Naxis parameter is matched, returns card offset in record.
      std::string::size_type parseNaxis(const std::string& record,
                                   std::string::size_type card_of_record = 0);
      //! Check the Naxis list, ltl::FitsException on error.
      void checkNaxis() throw(FitsException);
      //@}

      /*! \name Parse Card String
        These 2 functions parse a 80 char line card into an ltl::FitsCard.
      */
      //@{
      //! Parse 80 char line into an ltl::FitsCard object.
      FitsCard* parseCard(const std::string& card) const
         throw(FitsException);
      //! Return the comment of a 80 char line card remainder.
      std::string getCardComment(const std::string& teststring) const
         throw(FitsException);
      //@}

      //! Parse and add a commentstring to its list.
      void parseCommentToCardList(const std::string& keyword, const std::string& comment);
      //! Add a commentary card to the correct list.
      void addCommentCard(FitsCard* cardptr) throw(FitsException);

      //! Just split 80 char card into keyword + rest and push it on the junk list.
      void addJunkCard(const std::string& native_card);
      //! Add a suitable trimmed junk card.
      void addJunkCard(const std::string& keyword, const std::string& comment);
      //! Add a preexisting ltl::FitsCard to junk list.
      void addJunkCard(FitsCard* cardptr);

      //! Add a value holding card to its proper list.
      void addValueCard(FitsCard* cardptr) throw(FitsException);

      //! Clear a whole list of cards.
      void clearCardList(std::list< FitsCard * > & the_list);

      //! Erase first ltl::FitsCard mathcing \e keyword from \e the_list.
      int eraseCardFromList(const std::string& keyword, std::list< FitsCard * > & the_list);
      //! Erase all cards matching \e keyword from the ltl::FitsHeader::junk_ list.
      int eraseCardsFromJunkList(const std::string& keyword);

      //! Return pointer to card matching \e keyword in \e the_list.
      FitsCard* findCardInList(const std::string& keyword,
                               const std::list< FitsCard * > & the_list) const;
      //! Return pointer to first card matching \e keyword on any non commentary list.
      FitsCard* getValueCard(const std::string& keyword) const
         throw(FitsException);

      //! Return string holding new line broken comments of a whole list.
      std::string writeCommentsOfList(const std::list< FitsCard * > & the_list) const;

      //! Return string holding FITS formatted cards of a list.
      std::string writeCardsOfList(const std::list< FitsCard * > & the_list) const;

      //! Copy header from \e other.
      void copy(const FitsHeader& other);
      //! Copy list of ltl::FitsCard
      void copyCardList(std::list< FitsCard * > & dest_list,
                        const std::list< FitsCard * > & src_list);

      //! Test if region complies with FITS file geometry.
      void testRegion(const util::Region& testreg) const
         throw(FitsException);
};

}

/***********************************************************
 * Doxygen documentation block starts here
 **********************************************************/

/*! \fn void ltl::FitsHeader::addValueCard(FitsCard* cardptr)
  If the card has a bad syntax it's put on the ltl::FitsHeader::junk_ list.
  \throw ltl::FitsException on error.
*/

/*! \fn std::string ltl::FitsHeader::adjustKeyword(std::string keyword) const;
  Trailing blanks are significant for FITS keywords.
  If you want to add missing trailing blanks or
  simply trim too long keywords you may want to use this method.
*/

/*! \fn int ltl::FitsHeader::eraseCard(const std::string& keyword)
  Returns 0 on success, -1 on \e keyword not found.
  \throw ltl::FitsException on forbidden erasures.
*/

/*! \fn int ltl::FitsHeader::eraseCardFromList(const std::string& keyword,
  std::list< FitsCard * > & the_list)
  Return -1 if \e keyword not found, otherwise 0.
*/

/*! \fn int ltl::FitsHeader::eraseCardsFromJunkList(const std::string& keyword)
  Return the number of erased cards.
*/

/*! \fn FitsCard* ltl::FitsHeader::findCardInList(const std::string& keyword,
  const std::list< FitsCard * > & the_list) const
  Return NULL if no card is found.
*/

/*! \fn FitsCard* ltl::FitsHeader::getValueCard(const std::string& keyword) const
  \throw ltl::FitsException on error, i.e. no matching card found.
*/

/*! \fn std::string::size_type ltl::FitsHeader::parseMandatory(const std::string& record)

  Return offset to next non-mandatory card within record.
  \throw ltl::FitsException on error.
*/

/*! \fn std::string::size_type ltl::FitsHeader::parseRecord(const std::string& record,
  const bool alljunk = false)

  Returns the length of the record or 0 if "END" was found.
  \throw ltl::FitsException on error.
*/

/*! \fn void ltl::FitsHeader::parseCommentToCardList(const std::string& keyword,
  const std::string& comment)

  Can be long, will be broken into lines)
*/

/*! \var  ltl::FitsHeader::ArrayList ltl::FitsHeader::array_
  These are BSCALE, BZERO, BUNIT, BLANK,
  CTYPEn, CRPIXn, CRVALn, CRDELTn, CROTAn,
  DATAMIN and DATAMAX.
*/

/*! \var  ltl::FitsHeader::ObserveList ltl::FitsHeader::observe_
  These are AUTHOR, BLOCKED, DATE, DATE-OBS, DATExxxx, 
  EQUINOX, EPOCH, INSTRUME, OBJECT, OBSERVER, ORIGIN,
  REFERENC and TELESCOP.
  EPOCH will be copied to EQUINOX if EQUINOX not present.
*/

#endif

