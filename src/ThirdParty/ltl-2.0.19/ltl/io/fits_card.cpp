/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_card.cpp 558 2015-03-11 18:17:04Z cag $
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

#include <ltl/io/fits_card.h>
#include <cctype>
#include <sstream>

using util::StringFactory;
using util::FitsDate;
using util::UTDateException;
using util::StringException;
using std::istringstream;

namespace ltl {

// ------------------------------------------------------------------------
// FitsCard implementations

// method to check syntax of keyword, reserved? ...
// sorry for now, but the syntax check still is dirty
// if there is time left I will put it in an extra class
void FitsCard::check_syntax() throw(FitsException)
{
   // syntax check

   // check keyword syntax
   const string::size_type keywordstart = keyword_.find_first_not_of(' ');
   // keyword is blank? ok!
   if(keywordstart != keyword_.npos)
   {
      if(keywordstart > 0)
         throw FitsException("keyword '" + keyword_ + "' has leading blanks");
      const string::size_type firstblank = keyword_.find_first_of(' ');
      const string::size_type keywordend = keyword_.find_last_not_of(' ');
      if( (firstblank != keyword_.npos) && (firstblank < keywordend) )
         throw FitsException("keyword '" + keyword_ + "' has embedded blanks");
      // check on illegal chars
      for(string::size_type i=0; i < keywordend; ++i)
         if( ((keyword_[i] < 0x30) || (keyword_[i] > 0x39)) &&
             ((keyword_[i] < 0x41) || (keyword_[i] > 0x5A)) &&
             (keyword_[i] != 0x5F) && (keyword_[i] != 0x2D) )
         {
            const string exceptiontext = "keyword '" + keyword_ +
               "' has illegal characters";
            // + StringFactory::toString(int(i), 2);
            throw FitsException(exceptiontext);
         }
   }

   FitsKnownList::const_iterator knownIter = KNOWN.begin();
   const FitsKnownList::const_iterator knownEnd = KNOWN.end();

   // search reserved list
   // should be put in a stl container
   while(knownIter != knownEnd)
   {
      const string teststring = string( (*knownIter).WORD );
      const string::size_type testlength = teststring.length();
      if(keyword_.substr(0, testlength) == teststring)
         // handle CDn, PVn, PSn, PCn, ...
         // do not identify if keyword has no digit following
         if( testlength !=2 || (testlength == 2 && isdigit(keyword_[2])) )
            break;
      
      ++knownIter;
   }
   const FitsKnownList::const_iterator reservedIter = knownIter;

   if(reservedIter != knownEnd)    // it is a reserved key
   {

      what_res.id = (*reservedIter).IDENT; // set identifier

      // now the real nasty things

      // check if type is ok
      switch ( (*reservedIter).TYPE )
      {
         case EMPTY:
            if(!isComment())
               throw FitsException("no value allowed");
            break;
         case STRING:
            getString();
            break;
         case BOOL:
            getBool();
            break;
         case INT:
            getInt();
            break;
         case FLOAT:
            getFloat();
            break;
         case COMPLEX:
            getComplex();
            break;
      }

      // all mandatory must have right fixed setting
      // (END may be not fixed all others must)
      // other reserved keys may have fixed or not
      if( (*reservedIter).FIXED && !fixed_)
         throw FitsException("value must use fixed format");

      // check on function of reserved key
      switch ( (*reservedIter).FUNCTION )
      {
         case MANDATORY:
            what_res.mandatory = true;
            break;
         case ARRAY:
            what_res.array = true;
            break;
         case DATE:
         {
            try
            {
               FitsDate((*this).getString(), verbose_);
            }
            catch(util::UTDateException& utde)
            {
               throw FitsException(utde.what());
            }
            what_res.date = true;
            break;
         }
         case COMMENTARY:
            what_res.commentary = true;
            break;
         case EXTENSION:
            what_res.extension = true;
            break;
         case OTHER:
            what_res.other = true;
            break;
      }
   }
   // it is not on the list => defaults are ok
}

// mother: default card is of no specific type
bool FitsCard::isComment() const
{
   return false;
}

bool FitsJunkCard::isComment() const
{
   return true;
}

bool FitsCommentCard::isComment() const
{
   return true;
}


// constructors:
FitsCard::FitsCard(const string& key, const bool fix, const string& com, const int verbose)
   : keyword_(key), fixed_(fix), comment_(com), verbose_(verbose), what_res()
{
   if( keyword_.size() != PH_C.KEYWORD_LENGTH )
      throw FitsException("keyword must be 8 chars");
}

FitsJunkCard::FitsJunkCard(const string& key, const string& com)
   : FitsCard(key, false, com)
{ }
FitsJunkCard::FitsJunkCard(const FitsJunkCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_)
{ }

FitsCommentCard::FitsCommentCard(const string& key, const string& com)
   : FitsCard(key, false, com)
{
   check_syntax();
}
FitsCommentCard::FitsCommentCard(const FitsCommentCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_)
{
   check_syntax();
}

FitsStringCard::FitsStringCard( const string& key, const string val,
                                const bool fix, const string com, const int verbose )
   : FitsCard(key, fix, com, verbose), value_(val)
{
   check_syntax();
}
FitsStringCard::FitsStringCard(const FitsStringCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_, other.verbose_), value_(other.value_)
{
   check_syntax();
}

FitsBoolCard::FitsBoolCard( const string& key, const bool val,
                            const bool fix, const string com )
   : FitsCard(key, fix, com), value_(val)
{
   check_syntax();
}
FitsBoolCard::FitsBoolCard(const FitsBoolCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_), value_(other.value_)
{
   check_syntax();
}

FitsIntCard::FitsIntCard(const string& key, const long val,
                         const bool fix, const string com )
   : FitsCard(key, fix, com), value_(val)
{
   check_syntax();
}
FitsIntCard::FitsIntCard(const FitsIntCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_), value_(other.value_)
{
   check_syntax();
}

FitsFloatCard::FitsFloatCard(const string& key, const double val,
                             const bool fix, const string com )
   : FitsCard(key, fix, com), value_(val)
{
   check_syntax();
}
FitsFloatCard::FitsFloatCard(const FitsFloatCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_), value_(other.value_)
{
   check_syntax();
}

FitsComplexCard::FitsComplexCard(const string& key,
                                 const complex<double> val,
                                 const bool fix, const string com )
   : FitsCard(key, fix, com), value_(val)
{
   check_syntax();
}

FitsComplexCard::FitsComplexCard(const FitsComplexCard& other)
   : FitsCard(other.keyword_, other.fixed_, other.comment_), value_(other.value_)
{
   check_syntax();
}


// make clone
FitsCard* FitsJunkCard::clone()
{
   return new FitsJunkCard(*this);
}

FitsCard* FitsCommentCard::clone()
{
   return new FitsCommentCard(*this);
}

FitsCard* FitsStringCard::clone()
{
   return new FitsStringCard(*this);
}

FitsCard* FitsBoolCard::clone()
{
   return new FitsBoolCard(*this);
}

FitsCard* FitsIntCard::clone()
{
   return new FitsIntCard(*this);
}

FitsCard* FitsFloatCard::clone()
{
   return new FitsFloatCard(*this);
}

FitsCard* FitsComplexCard::clone()
{
   return new FitsComplexCard(*this);
}

// return keyword string
string FitsCard::getKeyword() const
{
   return keyword_;
}

// get value (default is no value => exception)
string FitsCard::getString() const
{
   throw FitsException("value of '" + keyword_ + "' is no valid string");
}
string FitsStringCard::getString() const
{
   return value_;
}

bool FitsCard::getBool() const
{
   throw FitsException("value of '" + keyword_ + "' is no valid boolian");
}
bool FitsBoolCard::getBool() const
{
   return value_;
}

long FitsCard::getInt() const
{
   throw FitsException("value of '" + keyword_ + "' is no valid integer");
}
long FitsIntCard::getInt() const
{
   return value_;
}

double FitsCard::getFloat() const
{
   throw FitsException("value of '" + keyword_ +
                       "' is no valid floating point number");
}
double FitsIntCard::getFloat() const
{
   return double(value_);
}
double FitsFloatCard::getFloat() const
{
   return value_;
}


complex<double> FitsCard::getComplex() const
{
   throw FitsException("value of '" + keyword_ +
                       "' is no valid complex number");
}
complex<double> FitsComplexCard::getComplex() const
{
   return value_;
}

bool FitsCard::isFixed() const
{
   return fixed_;
}

string FitsCard::getComment() const
{
   return comment_;
}

bool FitsCard::isReserved() const
{
   return (what_res.mandatory || what_res.array || what_res.date ||
           what_res.commentary || what_res.extension || what_res.other);
}

bool FitsCard::isMandatory() const
{
   return what_res.mandatory;
}

bool FitsCard::isArray() const
{
   return what_res.array;
}

bool FitsCard::isDate() const
{
   return what_res.date;
}

bool FitsCard::isCommentary() const
{
   return what_res.commentary;
}

bool FitsCard::isExtension() const
{
   return what_res.extension;
}

bool FitsCard::isOther() const
{
   return what_res.other;
}

reserved_keys_ident FitsCard::what_id() const
{
   return what_res.id;
}

bool FitsCard::isJunk() const
{
   return ( isComment() && ( !isCommentary() ) );
}


// get value as a string
string FitsStringCard::toString() const
{
   return value_;
}

string FitsBoolCard::toString() const
{
   ostringstream tmp;
   tmp << value_;
   return tmp.str();
}

string FitsIntCard::toString() const
{
   ostringstream tmp;
   tmp << value_;
   return tmp.str();
}

string FitsFloatCard::toString() const
{
   ostringstream tmp;
   tmp << value_;
   return tmp.str();
}

string FitsComplexCard::toString() const
{
   ostringstream tmp;
   tmp << value_;
   return tmp.str();
}

string FitsJunkCard::toString() const
{
   return comment_;
}

string FitsCommentCard::toString() const
{
   return comment_;
}


// mother: write card for header
string FitsCard::writeCard() const
{
   string cardstring = keyword_ + comment_;
   cardstring.resize(PH_C.CARD_LENGTH, ' ');
   return cardstring;
}

// same for daughters where default is insufficient
string FitsStringCard::writeCard() const
{
   string cardstring = keyword_ + "= \'" + value_;
   const string::size_type cardlength = cardstring.length();
   if( cardlength > string::size_type(PH_C.CARD_LENGTH - 1) )
      // string already too long?
      cardstring.resize(string::size_type(PH_C.CARD_LENGTH - 1), ' ');
   else
      // for fixed format strings => get minimum length by adding trailing blanks
      if(fixed_ && (cardlength < PH_C.FIXED_STRING_MIN_END) )
         cardstring.resize(PH_C.FIXED_STRING_MIN_END, ' ');
   cardstring += '\''; // closing quote
   if(comment_ != "")
   {
      // if possible add comment after column 30
      if( (cardstring.size() < PH_C.FIXED_VALUE_END) &&
          ( (comment_.size() + 3) < (PH_C.CARD_LENGTH - PH_C.FIXED_VALUE_END)) )
         cardstring.resize(string::size_type(PH_C.FIXED_VALUE_END), ' ');
      cardstring += " / " + comment_;
   }
   cardstring.resize(PH_C.CARD_LENGTH, ' ');
   return cardstring;
}

string FitsBoolCard::writeCard() const
{
   string cardstring = keyword_ + "= ";
   if(fixed_)
      cardstring.append(19, ' ');
   if(value_)
      cardstring += 'T';
   else
      cardstring += 'F';
   if(comment_ != "")
      cardstring += " / " + comment_;
   cardstring.resize(PH_C.CARD_LENGTH, ' ');
   return cardstring;
}

string FitsIntCard::writeCard() const
{
   string cardstring=keyword_ + "= ";
   try
   {
      if(fixed_)
         cardstring += StringFactory::toString(
            value_, PH_C.FIXED_VALUE_END - PH_C.VALUE_MIN_START);
      else
      {
         string valuestring = StringFactory::toString(value_);
         const string::size_type valuestart = valuestring.find_first_not_of(' ');
         valuestring.erase(string::size_type(0), valuestart);
         cardstring += valuestring;
      }
      if(comment_ != "")
         cardstring += " / " + comment_;
      cardstring.resize(PH_C.CARD_LENGTH, ' ');
   }
   catch(util::StringException& stre)
   {
      throw FitsException(stre.what());
   }

   return cardstring;
}

string FitsFloatCard::writeCard() const
{
   string cardstring = keyword_ + "= ";
   try
   {
      if(fixed_)
      {
         string valuestring = StringFactory::toString(
            value_, PH_C.FIXED_VALUE_END - PH_C.VALUE_MIN_START );
         const string::size_type exponent = valuestring.find_first_of('e');
         if(exponent != valuestring.npos)
            valuestring.replace(exponent, 1, "E");
         cardstring += valuestring;
      }
      else
      {
         string valuestring = StringFactory::toString(value_);
         const string::size_type valuestart = valuestring.find_first_not_of(' ');
         valuestring.erase(string::size_type(0), valuestart);
         cardstring += valuestring;
      }
      if(comment_ != "")
         cardstring += " / " + comment_;
      cardstring.resize(PH_C.CARD_LENGTH, ' ');
   }
   catch(util::StringException& stre)
   {
      throw FitsException(stre.what());
   }
   return cardstring;
}

string FitsComplexCard::writeCard() const
{
   string cardstring = keyword_ + "= ";
   // sorry ... still missing
   cardstring +=
      string(" \'sorry ... still not implemented complex card write\'");

   return cardstring;
}

}
