/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: stringfac.cpp 526 2013-11-26 17:04:46Z montefra $
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

#include <ltl/util/stringfac.h>
#include <cstdio>

namespace util {

const string StringFactory::factory_text =
   string("StringFactory: ");
const string StringFactory::error_text =
   string(" is too wide\ncannot create string with preset width.");

/*!
 * set the _floatfield_ format flag in `ostringstream` `s`
 */
void StringFactory::changeFormat(ostringstream& s, const char format)
{
	if(format == 'd') return;
	else if(format == 'f') s << std::fixed;
	else if(format == 'e') s << std::scientific;
	else throw StringException(factory_text + "unrecognised format identifier");
}

string StringFactory::toString(const double xd, const int width,
                               const int precision, const char format)
{
   return StringFactory::floatToString(xd, width, precision, format);
}

string StringFactory::toString(const float xd, const int width,
                               const int precision, const char format)
{
   return StringFactory::floatToString(xd, width, precision, format);
}

string StringFactory::toString(const long xd, const int width)
{
   return StringFactory::intToString(xd, width);
}

string StringFactory::toString(const int xd, const int width)
{
   return StringFactory::intToString(xd, width);
}

string StringFactory::toMinSecString(double xd, const int prec)
{
   char result[256];
   bool neg = false;

   if(xd < 0.0) {
      xd *= -1.0;
      neg = true;
   }

   const int deg = int(xd);
   const int min = int( 60.0 * fmod(xd, 1.0) );
   const double sec  = 60.0 * fmod(xd * 60.0, 1.0);
#ifdef HAVE_SNPRINTF
   if( neg )
      snprintf( result, 256, "-%02d:%02d:%0*.*f", deg, min,
                (prec>0)?(3+prec):2, prec, sec );
   else
      snprintf( result, 256, "%02d:%02d:%0*.*f", deg, min,
                (prec>0)?(3+prec):2, prec, sec );
#else
   if( neg )
      sprintf( result, "-%02d:%02d:%0*.*f", deg, min,
               (prec>0)?(3+prec):2, prec, sec );
   else
      sprintf( result, "%02d:%02d:%0*.*f", deg, min,
               (prec>0)?(3+prec):2, prec, sec );
#endif
   return string(result);
}

double StringFactory::toDouble( string xs)
{
   string::size_type col1, col2;
   double sign = 1.0;

   string s = xs.substr(0,1);
   if( s == "-" ) {
      sign = -1.0;
      xs = xs.erase( 0, 1 ); // Remove leading '-'
   }      
   
   col1 = xs.find_first_of( ":", 0 );
   if( col1 == string::npos )
      return sign * atof( xs.c_str() );
   col2 = xs.find_first_of( ":", col1+1 );
   if( col2 == string::npos )
      throw StringException( factory_text + xs +
                             string(" is ambiguous.") );
   const double deg = atof( xs.substr( 0, col1 ).c_str() );
   const double min = atof( xs.substr( col1+1, col2-col1 ).c_str() );
   const double sec = atof( xs.substr( col2+1 ).c_str() );
   return ( sign * ( deg + ( (min + (sec / 60.0) ) / 60.0 ) ) );
}

}

