/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: utdate.h 558 2015-03-11 18:17:04Z cag $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
 *                         Claus A. Goessl <cag@usm.uni-muenchen.de>
 *                         Arno Riffeser <arri@usm.uni-muenchen.de>
 *                         Jan Snigula <snigula@usm.uni-muenchen.de>
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

#ifndef __UTDATE_H__
#define __UTDATE_H__

#include <ltl/config.h>
#include <ltl/util/stringfac.h>

#include <cmath>

#include <string>
#include <iostream>
#include <iomanip>

#ifdef HAVE_SSTREAM
#ifdef HAVE_IOS

#define UTIL_USING_SSTREAM
#include <sstream>
using std::stringstream;
using std::ios_base;
using std::ends;

#endif
#endif

#ifndef UTIL_USING_SSTREAM

#ifdef HAVE_STRSTREAM
#include <strstream>
using std::strstream;
using std::ios;
using std::ends;

#else
#error <sstream> or <strstream> needed!
#endif

#endif

#ifdef TM_IN_SYS_TIME
#  include <sys/time.h>
#else
#  include <ctime>
#endif

#include <ltl/util/u_exception.h>

using std::string;
using std::setprecision;

namespace util {

//! Representing UT as a wrapped time_t mother class.
/*! 
  All output is UTC,
  don't get confused!
  The Time is kept in util::UTDate::ut_date_" and always is UTC.
  All constructors are built to get UTC.

  This class works for times between Jan. 1st 1970 (start of time_t epoch) and
  Dec. 31st 2037 due to overrun of 4 byte time_t epoch (in Jan. 2038).
  \throw util::UTDateException are thrown on error.
*/
class UTDate
{
   protected:
      char* toCString()const;
      time_t ut_date_;
      time_t mkuttime(struct tm* ut_tm);
      time_t mkdcftime(struct tm* dcf_tm);
   public:
      //! Get actual UT (now).
      UTDate() : ut_date_(::time(NULL))
      { }
      //! Assume \e init_date to hold UT.
      UTDate( const time_t& init_date ) : ut_date_(init_date)
      { }

      //! get internal time_t
      time_t time()const;

      //! Return UTC date string.
      string toString()const;
};

//! Convert from and to DCF-77 type string.
class DCFDate : public UTDate
{
   public:
      //! Get actual UT.
      DCFDate() : UTDate()
      { }
      //! Initialise from util::UTDate or heirs.
      DCFDate( const UTDate& d ) : UTDate(d)
      { }
      //! Interpret DCF-77 time string.
      DCFDate(const string& init_date);
      //! Return DCF-77 date string.
      string toString()const;
};

//! Convert from and to FITS DATE type string.
class FitsDate : public UTDate
{
      //! Boolean indicating use of deprecated dd/mm/yy format.
      bool shortformat_;
   public:
      //! Get actual UT.
      FitsDate() : UTDate(), shortformat_(false)
      { }
      // Initialise from util::UTDate or heirs.
      FitsDate( const UTDate& d) : UTDate(d), shortformat_(false)
      { }
      //! Interpret FITS DATE string (already is UTC)
      FitsDate( const string& init_date, const int verbose=0);

      //! Return UTC string in FITS DATE format.
      string toString()const;
};

//! Convert from and to Julian date.
class JulDate : public UTDate
{
   protected:
      //! Julian date at Jan. 1st 1970, 0.00
      static const double juldate_on_epoch_;    //=2440587.5;
      //! 24 * 60 * 60
      static const double seconds_of_day_;      //=86400.0;
      //! Julian date at Dec. 31st 2037, 24.00
      static const double juldate_end_epoch_;   //=2465424.5;
      time_t toTime_t(const double& init_date)const;
      time_t toTime_t(const string& init_date)const;
   public:
      //! Get actual UT.
      JulDate() : UTDate()
      { }
      //! Initialise from util::UTDate or heirs.
      JulDate( const UTDate& d) : UTDate(d)
      { }
      //! Interpret double as Julian Date.
      JulDate( const double init_date)
      {
         ut_date_ = toTime_t(init_date);
      }
      //! Interpret string as Julian Date.
      JulDate( const string& init_date)
      {
         ut_date_ = toTime_t(init_date);
      }

#ifdef HAVE_SSTREAM
#ifdef HAVE_IOS
#define UTIL_USING_SSTREAM
      //! Return Julian date string.
      string toString( const int prec=15 )const;
#endif
#endif
#ifdef UTIL_USING_SSTREAM
#undef UTIL_USING_SSTREAM
#else
      //! Return Julian date string.
      string toString( const int prec=8 )const;
#endif
      //! Return Julian date double.
      double toDouble()const;
};

class UTTime : public UTDate
{
   public:
      UTTime() : UTDate()
      { }
      UTTime( const UTDate& d ) : UTDate(d)
      { }
      UTTime( const string& t );

      string toString()const;
};

class LSTime : public UTDate
{
      double lon_;

      double calc();

   public:
      LSTime( double lon=0. ) : UTDate(), lon_(lon)
      { }
      LSTime( const UTDate& d, double lon=0. ) : UTDate(d), lon_(lon)
      { }
     
      string toString( const int prec=0 ) const;
      double toHour() const;
      double toDeg() const;
};

class GSTime : public LSTime
{

   public:
      GSTime() : LSTime(0.)
      { }
     
};

}

#ifdef UTIL_USING_SSTREAM
#undef UTIL_USING_SSTREAM
#endif

#endif
