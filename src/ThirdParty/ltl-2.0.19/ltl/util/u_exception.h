/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: u_exception.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __UTIL_U_EXCEPTION__
#define __UTIL_U_EXCEPTION__

#include <ltl/config.h>

#include <string>
#include <exception>

using std::string;
using std::exception;

namespace util {

//! Standard exception for namespace util methods.
class UException : public exception
{
   public:
      UException( const string& what )
            : whatStr_(what)
      { }

      UException( const char* what )
            : whatStr_(what)
      { }

      virtual ~UException() throw()
      { }

      virtual const char *what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

//! Derived exception for Date conversion errors.
class UTDateException : public UException
{
   public:
      UTDateException( const string& what )
            : UException(what)
      { }

      UTDateException( const char* what )
            : UException(what)
      { }

      virtual ~UTDateException() throw()
      { }
};

//! Derived exception for impossible string formatting requests.
class StringException : public UException
{
   public:
      StringException( const string& what )
            : UException(what)
      { }

      StringException( const char* what )
            : UException(what)
      { }

      virtual ~StringException() throw()
      { }
};

}

#endif
