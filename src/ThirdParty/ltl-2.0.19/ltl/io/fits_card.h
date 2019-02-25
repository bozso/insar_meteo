/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_card.h 558 2015-03-11 18:17:04Z cag $
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

#ifndef __FITS_CARD_H__
#define __FITS_CARD_H__

#include <ltl/config.h>

#include <ltl/util/utdate.h>
#include <ltl/util/stringfac.h>
#include <ltl/misc/exceptions.h>
#include <ltl/io/fits_const.h>

#include <string>
#include <complex>

using std::complex;

namespace ltl {

//! To extract some type information of reserved keys from a ltl::FitsCard.
struct what_reserved
{
      what_reserved() :
         mandatory(false), array(false),
         date(false), commentary(false),
         extension(false), other(false),
         id(none)
      { }

      what_reserved(const what_reserved& init) :
         mandatory(init.mandatory), array(init.array),
         date(init.date), commentary(init.commentary),
         extension(init.extension), other(init.other),
         id(init.id)
      { }

      bool mandatory : 1;  // SIMPLE, BITPIX, NAXIS...
      bool array : 1;      // BSCALE, BZERO, ...
      bool date : 1;       // DATE, DATE-OBS, DATExxxx
      bool commentary : 1; // COMMENT, HISTORY, BLANK
      bool extension : 1;  // XTENSION, ...
      bool other : 1;      // not one of above but still reserved
      reserved_keys_ident id;
};

/*! \struct what_reserved
  One reserved could even be of more types
  but the parser then has to be modified to check for it.
  If further classification is needed, the structure has to be
  expanded and an ltl::FitsCard method for identification added.
*/

//! Single FITS card mother.
class FitsCard
{
   private:
      FitsCard()
         : fixed_(false), verbose_(0)
      { }

      FitsCard(const FitsCard&)
         : fixed_(false), verbose_(0)
      { }

   protected:
      const string keyword_;
      const bool fixed_;
      const string comment_;
      const int verbose_;

      static const FitsPhysical PH_C; // for FITS physical constants

      // boolians to identify (and classify) reserved keywords
      what_reserved what_res;

      // static const struct that holds the information about
      // any reserved keyword is only needed for syntax check
      static const FitsKnownList KNOWN;

      void check_syntax()
         throw(FitsException); // for intensive syntax check
      virtual bool isComment() const;

   public:
      FitsCard(const string& key, const bool fix, const string& com, const int verbose = 0);
      virtual ~FitsCard()
      { }

      virtual FitsCard * clone() = 0;
      string getKeyword() const;

      // get value (default is no value => exception)
      virtual string getString() const;
      virtual bool getBool() const;
      virtual long getInt() const;
      virtual double getFloat() const;
      virtual complex<double> getComplex() const;

      bool isFixed() const;
      string getComment() const;
      bool isReserved() const;
      bool isMandatory() const;
      bool isArray() const;
      bool isDate() const;
      bool isCommentary() const;
      bool isExtension() const;
      bool isOther() const;

      reserved_keys_ident what_id() const;
      bool isJunk() const;

      virtual string writeCard() const;

      virtual string toString() const = 0;
};

// single card daughters

//! Any card, whole value field is interpreted as a comment.
class FitsJunkCard : public FitsCard
{
      virtual bool isComment() const;

   public:
      FitsJunkCard(const string& key, const string& com);
      FitsJunkCard(const FitsJunkCard& other);

      virtual ~FitsJunkCard()
      { }

      virtual FitsCard* clone();

      virtual string toString() const;
};

//! For valid Comment style card (no value, known comment key).
class FitsCommentCard : public FitsCard
{
      virtual bool isComment() const;

   public:
      FitsCommentCard(const string& key, const string& com);
      FitsCommentCard(const FitsCommentCard& other);

      virtual ~FitsCommentCard()
      { }

      virtual FitsCard* clone();

      virtual string toString() const;
};

//! FITS string type card.
class FitsStringCard : public FitsCard
{
   protected:
      string value_;

   public:
      FitsStringCard( const string& key, const string val = "",
                      const bool fix = false, const string com = "", const int verbose = 0);
      FitsStringCard(const FitsStringCard& other);

      virtual ~FitsStringCard()
      { }

      virtual FitsCard* clone();

      virtual string getString() const;

      virtual string writeCard() const;

      virtual string toString() const;
};

//! FITS boolean type card.
class FitsBoolCard : public FitsCard
{
   protected:
      bool value_;

   public:
      FitsBoolCard( const string& key, const bool val = false,
                    const bool fix = false, const string com = "");
      FitsBoolCard(const FitsBoolCard& other);

      virtual ~FitsBoolCard()
      { }

      virtual FitsCard* clone();

      virtual bool getBool() const;

      virtual string writeCard() const;

      virtual string toString() const;
};

//! FITS integer type card.
class FitsIntCard : public FitsCard
{
   protected:
      long value_;

   public:
      FitsIntCard(const string& key, const long val = 0l,
                  const bool fix = false, const string com = "");

      FitsIntCard(const FitsIntCard& other);

      virtual ~FitsIntCard()
      { }

      virtual FitsCard* clone();

      virtual long getInt() const;
      virtual double getFloat() const;

      virtual string writeCard() const;

      virtual string toString() const;
};

//! FITS floating point type card (which is actually closer to IEEE double).
class FitsFloatCard : public FitsCard
{
   protected:
      double value_;

   public:
      FitsFloatCard(const string& key, const double val = 0.0,
                    const bool fix = false, const string com = "");
      FitsFloatCard(const FitsFloatCard& other);

      virtual ~FitsFloatCard()
      { }

      virtual FitsCard* clone();

      virtual double getFloat() const;

      virtual string writeCard() const;

      virtual string toString() const;
};

//! Prototype for future implementation: FITS Complex type.
class FitsComplexCard : public FitsCard
{
   protected:
      complex<double> value_;

   public:
      FitsComplexCard(const string& key,
                      const complex<double> val,
                      const bool fix = false, const string com = "");
      FitsComplexCard(const FitsComplexCard& other);

      virtual ~FitsComplexCard()
      { }

      virtual FitsCard* clone();

      virtual complex<double> getComplex() const;

      virtual string writeCard() const;

      virtual string toString() const;
};
/*! \class FitsComplexCard
  \warning Still lacks implementation!
  This class will not be parsed in ltl::FitsHeader up to now
  and it has no writeCard implementation in fits_card.cpp.
*/

}

#endif
