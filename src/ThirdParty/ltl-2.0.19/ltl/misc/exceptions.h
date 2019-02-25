/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: exceptions.h 491 2011-09-02 19:36:39Z drory $
* ---------------------------------------------------------------------
*
* Copyright (C)  Niv Drory <drory@mpe.mpg.de>
*                         Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#ifndef __LTL_EXCEPTIONS__
#define __LTL_EXCEPTIONS__

#include <ltl/config.h>

#include <exception>
#include <string>

namespace ltl {
using std::exception;
using std::string;

//! Exception indicating problems with LTL
class LTLException : public exception
{
   public:
      LTLException( const string& what )
            : whatStr_(what)
      { }

      LTLException( const char* what )
            : whatStr_(what)
      { }

      virtual ~LTLException() throw()
      { }

      virtual const char *what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

//! Exception indicating problems with ASCII I/O.
class IOException : public exception
{
   public:
      IOException( const string& what )
            : whatStr_(what)
      { }

      IOException( const char* what )
            : whatStr_(what)
      { }

      virtual ~IOException() throw()
      { }

      virtual const char *what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};


//! Exception for ltl::MArray<T ,N> range check errors.
class RangeException : public exception
{
   public:
      RangeException( const string& what )
            : whatStr_(what)
      { }

      RangeException( const char* what )
            : whatStr_(what)
      { }

      virtual ~RangeException() throw()
      { }

      virtual const char *what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

// catch it if you want to know what's not ok
// method what() gives a explanatory string
//! Exception indicating problems with FITS I/O.
class FitsException : public exception
{
   public:
      FitsException( const string & what )
            : whatStr_(what)
      { }

      FitsException( const char * what )
            : whatStr_(what)
      { }

      virtual ~FitsException() throw()
      { }

      virtual const char * what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

// catch it if you want to know what's not ok
// method what() gives a explanatory string
//! Exception indicating problems within Linear Algebra.
class LinearAlgebraException : public exception
{
   public:

      LinearAlgebraException( const string & what )
         : whatStr_(what)
      { }

      LinearAlgebraException( const char * what )
         : whatStr_(what)
      { }

      virtual ~LinearAlgebraException() throw()
      { }

      virtual const char * what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

//! Exception indicating a singular Matrix.
class SingularMatrixException : public LinearAlgebraException
{
   public:
      SingularMatrixException( const string & what )
         : LinearAlgebraException(what), whatStr_(what)
      { }

      SingularMatrixException( const char * what )
         :  LinearAlgebraException(what), whatStr_(what)
      { }

      virtual ~SingularMatrixException() throw()
      { }

      virtual const char * what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

//! Exception indicating a divergent iterative algorithm.
class DivergenceException : public LinearAlgebraException
{
   public:
      DivergenceException( const string & what )
         :  LinearAlgebraException(what), whatStr_(what)
      { }

      DivergenceException( const char * what )
         :  LinearAlgebraException(what), whatStr_(what)
      { }

      virtual ~DivergenceException() throw()
      { }

      virtual const char * what() const throw()
      {
         return whatStr_.c_str();
      }

   protected:
      string whatStr_;
};

}

#endif
