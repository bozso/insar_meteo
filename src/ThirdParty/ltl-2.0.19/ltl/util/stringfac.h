/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: stringfac.h 526 2013-11-26 17:04:46Z montefra $
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

#ifndef __STRINGFAC_H__
#define __STRINGFAC_H__

#include <ltl/config.h>

#include <string>
#include <iostream>
#include <iomanip>

#ifdef HAVE_SSTREAM
#ifdef HAVE_IOS

#define UTIL_USING_SSTREAM
#include <sstream>
using std::ostringstream;
using std::ios;
using std::setw;
using std::setprecision;
using std::setiosflags;

#endif
#endif

#ifndef UTIL_USING_SSTREAM

#ifdef HAVE_STRSTREAM
#include <strstream>
using std::ostrstream;
using std::ios;
using std::setw;
using std::setprecision;
using std::ends;
using std::setiosflags;

#else
#error <sstream> or <strstream> needed!
#endif

#endif

#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>

#include <ltl/util/u_exception.h>

using std::string;
// using std::cerr;
// using std::endl;

namespace util {

//! This class returns a number in the requested type.
/*! In case of string output this is a right adjusted string
  of selected length and (if possible) precision.
  The string is filled with leading blanks to match width.
*/
class StringFactory
{
   protected:
      static const string factory_text;
      static const string error_text;

      /*!
       * Convert float/double to string
       * \param format: char to select the _floatfield_ format flag
       *   - 'd': default floating-point notation
       *   - 'f': fixed-point notation
       *   - 'e': scientific notation
       */
      template<class T>
      inline static string floatToString(const T xd, const int width,
                                         const int precision, const char format);
      template<class T>
      inline static string intToString(const T xd, const int width);
      
      /*!
       * set the _floatfield_ format flag
       * \param s: `ostringstream` where to write
       * \param format: char to select the _floatfield_ format flag
       *   - 'd': default floating-point notation
       *   - 'f': fixed-point notation
       *   - 'e': scientific notation
       */
      void static changeFormat(ostringstream& s, const char format);

   public:
      /*!
       * Convert number to string
       * \param format: (for double and float only) char to select the _floatfield_ format flag
       *   - 'd': default floating-point notation
       *   - 'f': fixed-point notation
       *   - 'e': scientific notation
       */
      static string toString(const double xd,
                             const int width = DBL_DIG + 7,
                             const int precision = DBL_DIG,
                             const char format = 'd');
      static string toString(const float xd,
                             const int width = FLT_DIG + 7,
                             const int precision = FLT_DIG,
                             const char format = 'd');
      static string toString(const long xd,
                             const int width = ( (sizeof(long) * 8) / 3) + 1);
      static string toString(const int xd,
                             const int width = ( (sizeof(int) * 8) / 3) + 1);

      //! Return \e xd in dd:mm:ss[.sss] format.
      /*! Precision gives No second decimals. */
      static string toMinSecString(const double xd,
                                   const int precision = 0);

      //! Convert \e xs to integer number.
      static int toInt(const string& xs)
      {
         return atoi(xs.c_str());
      }
      //! Convert \e xs to long integer number.
      static long toLong(const string& xs)
      {
         return atol(xs.c_str());
      }
      //! Convert \e xs to float number.
      /*! dd:mm:ss[.sss] strings are automatically recognized. */
      static float toFloat(const string& xs)
      {
         return float(toDouble(xs));
      }
      //! Convert \e xs to double number.
      /*! dd:mm:ss[.sss] strings are automatically recognized. */
      static double toDouble(string xs);

};

#ifdef UTIL_USING_SSTREAM
template<class T>
string StringFactory::floatToString(const T xd,
                                    const int width, const int precision,
                                    const char format)
{
   string s_string;
   {
      ostringstream s_stream;
      changeFormat(s_stream, format);
      s_stream << setiosflags(ios::right | ios::adjustfield)
               << setw(width) << setprecision(precision) << xd;
      s_string = s_stream.str();
//       cerr << xd << ", width " << width << ", prec " << precision 
//            << ", gap " << s_string.length() - width << endl;
   }
   int gap = s_string.length() - width;
   if(gap > 0)
   {
      if(gap < precision)
      {
         ostringstream another;
         changeFormat(another, format);
         another << setiosflags(ios::right | ios::adjustfield)
                 << setw(width) << setprecision(precision - gap) << xd;
         s_string = another.str();
      }
      else
      {
         s_string += error_text;
         throw StringException(factory_text + s_string);
      }
   }
   return s_string;
}

template<class T>
string StringFactory::intToString(const T xd, const int width)
{
   string s_string;
   {
      ostringstream s_stream;
      s_stream << setiosflags(ios::right | ios::adjustfield)
               << setw(width) << xd;
      s_string = s_stream.str();
//       cerr << xd << ", width " << width
//            << ", gap " << s_string.length() - width << endl;
   }
   int gap = s_string.length() - width;
   if(gap > 0)
   {
      s_string += error_text;
      throw StringException(factory_text + s_string);
   }
   return s_string;
}

#undef UTIL_USING_SSTREAM
#else

// deprecated strstream version
template<class T>
string StringFactory::floatToString(const T xd,
                                    const int width, const int precision,
                                    const char format)
{
   string s_string;
   {
      ostrstream s_stream;
      s_stream.setf(ios::right, ios::adjustfield);
      s_stream << setw(width) << setprecision(precision) << xd << ends;
      s_string = s_stream.str();
      s_stream.freeze(0);
//       cerr << xd << ", width " << width << ", prec " << precision 
//            << ", gap " << s_string.length() - width << endl;
   }
   int gap = s_string.length() - width;
   if(gap > 0)
   {
      if(gap < precision)
      {
         ostrstream another;
         another.setf(ios::right, ios::adjustfield);
         another << setw(width) << setprecision(precision - gap) << xd << ends;
         s_string = another.str();
         another.freeze(0);
      }
      else
      {
         s_string += error_text;
         throw StringException(factory_text + s_string);
      }
   }
   return s_string;
}

template<class T>
string StringFactory::intToString(const T xd, const int width)
{
   string s_string;
   {
      ostrstream s_stream;
      s_stream.setf(ios::right, ios::adjustfield);
      s_stream << setw(width) << xd << ends;
      s_string = s_stream.str();
      s_stream.freeze(0);
//       cerr << xd << ", width " << width
//            << ", gap " << s_string.length() - width << endl;
   }
   int gap = s_string.length() - width;
   if(gap > 0)
   {
      s_string += error_text;
      throw StringException(factory_text + s_string);
   }
   return s_string;
}

#endif

}

#endif
