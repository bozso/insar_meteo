/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: utdate.cpp 558 2015-03-11 18:17:04Z cag $
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

// class for many time specific functions
// all output is converted to UTC

// valid for times between Jan. 1st 1970 and
// Dec. 31st 2037 due to overrun of epoche (Jan. 2038)
// Exception type UTDateException are thrown on error

#include <ltl/util/utdate.h>

using std::cerr;
using std::endl;
using std::setiosflags;
using std::ios;

namespace util {

const double JulDate::juldate_on_epoch_  = 2440587.5;
const double JulDate::seconds_of_day_    = 86400.0;
const double JulDate::juldate_end_epoch_ = 2465424.5;


// Date class methods

char * UTDate::toCString()const
{
   return asctime( gmtime(&ut_date_) );
}

time_t UTDate::mkuttime(struct tm* ut_tm)
{
   time_t loc_date = mktime(ut_tm);
   double offset = difftime(loc_date, mktime( gmtime(&loc_date) ));
   return ( loc_date + time_t(offset) );
}

time_t UTDate::mkdcftime(struct tm* dcf_tm)
{
   const time_t hour = 3600;
   return ( mkuttime(dcf_tm) - hour );
}

time_t UTDate::time()const
{
   return ut_date_;
}

string UTDate::toString()const
{
   string return_string = string(toCString());
   return_string.resize( return_string.length() - 1 );
   return return_string;
}


// DCFDate class methods

DCFDate::DCFDate( const string& init_date )
{
   // in case of syntax error prepare message
   const string syntax_error =
      "DCFDate: string '" + init_date +
      "' has wrong syntax\n(not \"D:DD.MM.YY;T:w;U:hh.mm.ss;    \")";
   // tm structure for conversion in time_t
   struct tm dcfdate =
      {
         0, 0, 0, 0, 0, 0, 0, 0, 0
      };
   // parse string
   if( (init_date.size() != 30) )
      throw UTDateException(syntax_error);
   dcfdate.tm_sec  = atoi( (init_date.substr(23, 2)).c_str() );
   dcfdate.tm_min  = atoi( (init_date.substr(20, 2)).c_str() );
   dcfdate.tm_hour = atoi( (init_date.substr(17, 2)).c_str() );
   dcfdate.tm_mday = atoi( (init_date.substr(2, 2)).c_str() );
   dcfdate.tm_mon  = atoi( (init_date.substr(5, 2)).c_str() ) - 1;
   dcfdate.tm_year = atoi( (init_date.substr(8, 2)).c_str() );
   if(dcfdate.tm_year < 38)
      dcfdate.tm_year += 100;
   if(init_date[28] == 'S')
      dcfdate.tm_isdst = 1;

   if(dcfdate.tm_year < 70)
      throw UTDateException("DCFDate: cannot handle dates before Jan. 1st 1970.");
   if(dcfdate.tm_year > 137)
      throw UTDateException("DCFDate: Cannot handle dates after Dec. 31st 2037.");
   ut_date_ = mkdcftime(&dcfdate);
   if(ut_date_ < 0)
   {
      const string exception_text = "DCFDate: Cannot convert date " + init_date;
      throw UTDateException(exception_text);
   }
}

string DCFDate::toString()const
{
   const size_t n = 32;
   const time_t hour = 3600;
   const char dcf_format_string[n] =
      {"D:%d.%m.%y;T: ;U:%H.%M.%S;    "};
   char date_string_buffer[n];
   time_t dcf_date=ut_date_ + hour;
   struct tm *dcfdateptr = gmtime(&dcf_date);
   strftime(date_string_buffer, n, dcf_format_string, dcfdateptr);
   date_string_buffer[13] = char(dcfdateptr->tm_wday) + '0';
   return string(date_string_buffer);
}


// FitsDate class methods

FitsDate::FitsDate( const string& init_date, const int verbose )
{
   // in case of syntax error prepare message
   const string syntax_error =
      "FitsDate: string '" + init_date +
      "' has wrong syntax\n(not DD/MM/YY or YYYY-MM-DD[Thh:mm:ss[.sss...]])";
   // tm structure for conversion in time_t
   struct tm fitsdate =
      {
         0, 0, 0, 0, 0, 0, 0, 0, 0
      };
   // get old format DD/MM/YY
   if(init_date.size() == 8U)
   {
      if( (init_date[2] != '/') || (init_date[5] != '/') )
         throw UTDateException(syntax_error);
      fitsdate.tm_mday = atoi((init_date.substr(0,2)).c_str());
      fitsdate.tm_mon  = atoi((init_date.substr(3,2)).c_str()) - 1;
      fitsdate.tm_year = atoi((init_date.substr(6,2)).c_str());
      if(fitsdate.tm_year < 38)
      {
         if(verbose)
            cerr << "FitsDate: Assuming date "
                 << (2000 + fitsdate.tm_year)
                 << ". Date has obsolete syntax." << endl;
         fitsdate.tm_year += 100;
      }
      shortformat_ = true;
   }
   else
   {
      // get new format YYYY-MM-DD
      if(init_date.size() == 10U)
      {
         if( (init_date[4] != '-') || (init_date[7] != '-') )
            throw UTDateException(syntax_error);
         shortformat_ = true;
      }
      else
      {
         // get new extended format
         if(init_date.size() > 18U)
         {
            if( (init_date[10] != 'T') || (init_date[13] != ':') || (init_date[16] != ':') )
               throw UTDateException(syntax_error);
            fitsdate.tm_sec  = atoi( (init_date.substr(17, 2)).c_str() );
            fitsdate.tm_min  = atoi( (init_date.substr(14, 2)).c_str() );
            fitsdate.tm_hour = atoi( (init_date.substr(11, 2)).c_str() );
            shortformat_ = false;
         }
         else
            throw UTDateException(syntax_error);
      }
      fitsdate.tm_mday = atoi( (init_date.substr(8, 2)).c_str() );
      fitsdate.tm_mon  = atoi( (init_date.substr(5, 2)).c_str() ) - 1;
      fitsdate.tm_year = atoi( (init_date.substr(0, 4)).c_str() ) - 1900;
   }
   if(fitsdate.tm_year < 70)
      throw UTDateException("FitsDate: cannot handle dates before Jan. 1st 1970.");
   if(fitsdate.tm_year > 137)
      throw UTDateException("FitsDate: Cannot handle dates after Dec. 31st 2037.");
   ut_date_ = mkuttime(&fitsdate);
   if(ut_date_ < 0)
   {
      const string exception_text = "FitsDate: Cannot convert date " + init_date;
      throw UTDateException(exception_text);
   }
}


string FitsDate::toString()const
{
   const size_t n = 32;
   const char fits_format_string[n] =
      {"%Y-%m-%dT%H:%M:%S"};
   const char short_format_string[n] =
      {"%Y-%m-%d"};
   // the old format string shall not be used any more even with old dates
   //const char old_fits_format_string[n] = {"%d/%m/%y"};
   char date_string_buffer[n];
   struct tm *fitsdateptr = gmtime(&ut_date_);
   strftime(date_string_buffer, n,
            // ((fitsdateptr->tm_year>=99) ? (fits_format_string) : (old_fits_format_string)),
            ( shortformat_ ? short_format_string : fits_format_string ),
            fitsdateptr);
   return string(date_string_buffer);
}


// JulDate class methods

time_t JulDate::toTime_t( const double& init_date )const
{
   if(init_date < juldate_on_epoch_)
      throw UTDateException("JulDate: cannot handle dates before Jan. 1st 1970");
   if(init_date > juldate_end_epoch_)
      throw UTDateException("JulDate: cannot handle dates after Dec. 31st 2037");
   return time_t( (init_date - juldate_on_epoch_) * seconds_of_day_ + 0.5 );
}

time_t JulDate::toTime_t( const string& init_date )const
{
   return toTime_t( atof(init_date.c_str()) );
}

#ifdef HAVE_SSTREAM
#ifdef HAVE_IOS
#define UTIL_USING_SSTREAM
string JulDate::toString( const int prec )const
{
   stringstream s_stream;
   string s_string;
   s_stream << setiosflags(ios::fixed | ios::floatfield)
            << setprecision(prec)
            << (toDouble()) << ends;
   s_stream >> s_string;
   return s_string;
}
#endif
#endif

#ifdef UTIL_USING_SSTREAM
#undef UTIL_USING_SSTREAM
#else
string JulDate::toString( const int prec )const
{
   strstream s_stream;
   string s_string;
   s_stream.setf(ios::fixed, ios::floatfield);
   s_stream.precision(prec);
   s_stream << (toDouble()) << ends;
   s_stream >> s_string;
   return s_string;
}
#endif

double JulDate::toDouble() const
{
   const double juldate_since_epoch =
      double(ut_date_) / seconds_of_day_;
   return (juldate_on_epoch_ + juldate_since_epoch);
}

UTTime::UTTime( const string& t ) : UTDate()
{ 
   // in case of syntax error prepare message
   const string syntax_error =
      "UTTime: string '" + t +
      "' has wrong syntax\n(not ##:##:##[.##])";

   struct tm *ct = gmtime( &ut_date_ );
   
   string::size_type col1, col2;
   
   col1 = t.find_first_of( ":", 0 );
   if( col1 == string::npos )
      throw UTDateException( syntax_error );
   
   col2 = t.find_first_of( ":", col1+1 );
   if( col2 == string::npos )
      throw UTDateException( syntax_error );
   
   ct->tm_hour = atoi( t.substr( 0, col1 ).c_str() );
   ct->tm_min  = atoi( t.substr( col1+1, col2-col1 ).c_str() );
   ct->tm_sec  = atoi( t.substr( col2+1 ).c_str() );
   
   ut_date_ = mkuttime(ct);
}

string UTTime::toString() const
{
   const size_t n = 16;
   const char format_string[n] =
      {"%H:%M:%S"};
   char time_string_buffer[n];
   struct tm *timeptr = gmtime(&ut_date_);
   strftime(time_string_buffer, n, format_string, timeptr);
   return string(time_string_buffer);
}

string LSTime::toString( const int prec ) const
{
   return StringFactory::toMinSecString( toHour(), prec );
}

double LSTime::toDeg() const
{
   const double jd = JulDate( *this ).toDouble();
   const double t = ( jd - 2451545.0) / 36525.;
   double gst_deg =  280.46061837 + ( 360.98564736629 * 36525. +
      ( 0.000387933 - t/38710000. ) * t ) * t; 
   const double f = fmod( gst_deg, 360. ) - lon_;

   return (f>0.) ? f : (f+360.);
}

double LSTime::toHour() const
{
   return toDeg()/15.;
}


}
