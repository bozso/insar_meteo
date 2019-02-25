/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: region.cpp 499 2011-12-16 23:41:55Z drory $
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



#include <ltl/util/region.h>

namespace util {

// implementation
Region::Region(const size_t N) :
   N_(N), start_(new int [2*N_]), end_(start_ + N_)
{
   for(size_t i = 0; i < N_; ++i)
   {
      start_[i] = 0;
      end_[i] = 0;
   }
}

Region::Region(const string& s, const size_t N) :
   N_(N), start_(new int [2*N_]), end_(start_ + N_)
{
   // evaluate N_ from string does not comply with int *const start_ and int *const end_
   // you may use next few lines of code to evaluate No dimonsions from string
   // if( N_ == 0 )
   // {
   //    ++N_;
   //    fostring::const_iterator s_i = s.begin; s_i != s.end(); ++s_i)
   //       if(*s_i == ',') ++N_;
   // }

   if( (s[0] != '[') || (s[s.length()-1] != ']') )
      throw(UException( string("Region string ") + s + string(" not within [ ].") ));

   string str = s.substr( 1, s.length()-2 );
   string token;
   string::size_type end = 1;

   // comma separated string of dim_i_start:dim_i_end
   for( size_t i = 1; i <= N_; ++i )
   {
      end = str.find_first_of( ":", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
         throw(UException( string("Missing startvalue endvalue separator ':' in ") + s ));

      const int startvalue = atoi( token.c_str() );
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         if(i < N_)
            throw(UException( string("Missing axes separator ',' in ") + s ));
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      const int endvalue = atoi( token.c_str() );
      setRange(i,
               ( (endvalue > startvalue) ? startvalue : endvalue ),
               ( (endvalue > startvalue) ? endvalue : startvalue )); // swap if s>e
   }
}

Region::Region(const Region& other) :
   N_(other.N_), start_(new int [2*N_]), end_(start_ + N_)
{
   setRegion(other);
}

int Region::getStart(const size_t i) const throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_get_dim);
   return start_[i-1];
}

int Region::getEnd(const size_t i) const throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_get_dim);
   return end_[i-1];
}

size_t Region::getLength() const throw(UException)
{
   size_t totlength = 1;
   for(size_t i = 0; i < N_; ++i)
      totlength *= abs(end_[i] - start_[i]) + 1;
   return totlength;
}

size_t Region::getLength(const size_t i) const throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_get_dim);
   return abs(end_[i-1] - start_[i-1]) + 1;
}

Region Region::getSlice(const size_t startdim, const size_t enddim)
   const throw(UException)
{
   if( (startdim > enddim) || (startdim < 1) || (enddim > N_))
      throw UException(error_get_dim);
   Region slice( size_t(1 + enddim - startdim) );
   for(size_t i = 0, j = startdim - 1; j < enddim; ++i, ++j)
   {
      slice.start_[i] = start_[j];
      slice.end_[i] = end_[j];
   }
   return slice;
}

void Region::setStart(const size_t i, const int s) throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_set_dim);
   start_[i-1] = s;
}

void Region::setEnd(const size_t i, const int e) throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_set_dim);
   end_[i-1] = e;
}

void Region::setRange(const size_t i, const int s, const int e)
   throw(UException)
{
   if( (i < 1) || (i > N_) )
      throw UException(error_set_dim);
   start_[i-1] = s;
   end_[i-1] = e;
}

void Region::setRegion(const Region& other) throw(UException)
{
   if( N_ != other.N_ )
      throw UException(error_set_dim);
   for(size_t i = 0; i < N_; ++i)
   {
      start_[i] = other.start_[i];
      end_[i] = other.end_[i];
   }
}

string Region::toString() const
{
   string s = "[";
   size_t i;
   char buf[255];
   for( i = 0; i < (N_-1); ++i )
   {
      snprintf( buf, sizeof(buf), "%d:%d,", start_[i], end_[i] );
      s += string(buf);
   }
   snprintf( buf, sizeof(buf), "%d:%d", start_[i], end_[i] );
   s += string(buf) + string("]");
   return s;
}

const string Region::error_get_dim =
   "request for region parameter of illegal dimension";
const string Region::error_set_dim = 
   "request to set region parameter of illegal dimension";

}
