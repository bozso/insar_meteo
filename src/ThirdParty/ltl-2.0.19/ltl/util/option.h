/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: option.h 499 2011-12-16 23:41:55Z drory $
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


#ifndef __UTIL_OPTION_H__
#define __UTIL_OPTION_H__

#include <ltl/config.h>

#include <string>
#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <cstdio>

#include <ltl/util/u_exception.h>
#include <ltl/util/region.h>

using std::string;
using std::ostream;

namespace util {

/*! \addtogroup util_options
*/
//@{

//! The base class for all options.
/*! 
  All data is stored as strings. Conversion (parsing) to the actual
  type happens in derived specializations.

  Constructing an option requires 4 (+1) parameters:
  \li  the option's name
  \li  the default value as a string
  \li  the help string to print for this option
  \li  the character representing the option in the command line '-x'
  if this char is 0, the option will be a long option with it's
  name used as the option string for the command line '--name'
  \li The constructors of the heirs have a further parameter
  which is a pointer to the option's type.
  The option will also write its value to that location
  if the pointer is not \c NULL. Useful for easily setting global parameters, or
  filling a struct with parameter values without having to manually loop over
  and read all program option's values.
*/
class Option
{
   public:
      //@{
      /*! ctors and dtor
       *  The parameters are
       *  \li \c name The long option name, to be used with --option-name (command line), or option-name = (config file).
       *  \li \c defaultVal string representation of default value
       *  \li \c usage option description and help text
       *  \li \c the short option character, for use with -c on the command line. Supply \c '\0' if no short option name is needed.
       */
      Option( const char* name, const char* defaultVal,
              const char* usage, const char cmd );
      Option( const string& name, const string& defaultVal,
              const string& usage, const char cmd );
      virtual ~Option() throw();
      //@}

      //! This function is the heart of the \c Option class: it parses the value of the option from the supplied \c string.
      virtual void setValue( const string& s ) throw (UException) = 0;

      //@{
      /*! Acess the value of the option. Usually subclasses will not implement all of these, in fact, mostly just one of these. */
      virtual int getInt() const
      {
         return 0;
      }
      virtual float getFloat() const
      {
         return 0.;
      }
      virtual string getString() const
      {
         return "";
      }
      virtual bool getBool() const
      {
         return false;
      }
      //@}

      //! Return the type name of the option as a string
      virtual string getTypeName() const
      {
         return "UNKNOWN";
      }

      /*! Return \c true if the option needs a value, \c false if it is a toggle-switch needing no value.
       *  On the command line, values are given as --long-option value, or -x value. In a config file,
       *  long-option = value. '#' can be used to delineate comments. See \c CommandLineReader and
       *  \c ConfigFileReader.
       */
      virtual bool needsValue() const
      {
         return true;
      }

      //! Return the (long) option name.
      const string& getName() const;
      //! Return the short option char.
            char    getCmdLineChar() const;
      //! Return the string representation of the default value.
      const string& getDefault() const;
      //! Return the help string.
      const string& getUsage() const;

      //! Return the option's value as a string.
      virtual string toString() const = 0;

   protected:
      string name_;        //!< long option name (e.g. --long-option or long-option=value)
      string default_;     //!< string representation of default value
      string usage_;       //!< help string
      char   cmdlinechar_; //!< short option char, for command line use
};

/*! \relates util::Option
     Pretty-print the option to \e os.
*/
ostream& operator<<( ostream& os, Option& op );

/*!
 * These are the specializations for various types and arrays of types.
 * Each of these implements overloads the parser function:
 * \code void setValue( const string& s ); \endcode
 * and the accessor function(s) returning the appropriate type.
 *
 * The constructors have a further parameter which is a pointer to the
 * option's (native) type. The option will also write its value to that location
 * if the pointer is not NULL; Useful for easily setting global parameters, or
 * filling a struct with parameter values without having to manually loop over
 * and read all program option's values.
 */
//@{
//! Hold an integer number.
class IntOption : public Option
{
   public:

      IntOption( const char* name, const char* defaultVal,
                 const char* usage, const char cmd, int *const storage=NULL );
      IntOption( const string& name, const string& defaultVal,
                 const string& usage, const char cmd, int *const storage=NULL );

      ~IntOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "INT";
      }

      void setValue( const string& s ) throw (UException);

      virtual int getInt() const;

      virtual string toString() const;

   protected:
      int value_;
      int *storage_;
};


//! Hold a single precision number.
class FloatOption : public Option
{
   public:

      FloatOption( const char* name, const char* defaultVal,
                   const char* usage, const char cmd,
                   float *const storage=NULL );
      FloatOption( const string& name, const string& defaultVal,
                   const string& usage, const char cmd,
                   float *const storage=NULL );

      ~FloatOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "FLOAT";
      }

      void setValue( const string& s ) throw (UException);

      virtual float getFloat() const;

      virtual string toString() const;

   protected:
      float value_;
      float *storage_;
};


//! Hold a double precision number.
class DoubleOption : public Option
{
   public:

      DoubleOption( const char* name, const char* defaultVal,
                    const char* usage, const char cmd,
                    double *const storage=NULL );
      DoubleOption( const string& name, const string& defaultVal,
                    const string& usage, const char cmd,
                    double *const storage=NULL );

      ~DoubleOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "DOUBLE";
      }

      void setValue( const string& s ) throw (UException);

      virtual float getDouble() const;

      virtual string toString() const;

   protected:
      double value_;
      double *storage_;
};

//! Hold a boolean. Default value is toggled when the option is given.
class BoolOption : public Option
{
   public:

      BoolOption( const char* name, const char* defaultVal,
                  const char* usage, const char cmd, bool *const storage=NULL );
      BoolOption( const string& name, const string& defaultVal,
                  const string& usage, const char cmd, bool *const storage=NULL );

      ~BoolOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "FLAG";
      }

      void setValue( const string& s ) throw (UException);

      virtual bool getBool() const;

      virtual string toString() const;

      virtual bool needsValue() const
      {
         return false;
      }

   protected:
      bool value_;
      bool *storage_;
};

//! Hold a string.
class StringOption : public Option
{
   public:

      StringOption( const char* name, const char* defaultVal,
                    const char* usage, const char cmd,
                    string *const storage=NULL );
      StringOption( const string& name, const string& defaultVal,
                    const string& usage, const char cmd,
                    string *const storage=NULL );

      ~StringOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "STRING";
      }

      void setValue( const string& s ) throw (UException);

      virtual string getString() const;

      virtual string toString() const;

   protected:
      string value_;
      string *storage_;
};



//! Hold an array of N floats.
class FloatArrayOption : public Option
{
   public:

      FloatArrayOption( const char* name, const char* defaultVal,
                        const char* usage, const char & cmd,
                        const int & N, float *const storage=NULL );
      FloatArrayOption( const string& name, const string& defaultVal,
                        const string& usage, const char & cmd,
                        const int & N, float *const storage=NULL );
      FloatArrayOption( const FloatArrayOption &other );

      ~FloatArrayOption() throw();

      virtual string getTypeName() const
      {
         return "FLOATARRAY";
      }

      void setValue( const string& s ) throw (UException);

      virtual float* getFloatArray() const;

      virtual string toString() const;

   protected:
      const int N_;
      float * const storage_;
      float * const value_;
};



//! Hold an array of N doubles.
class DoubleArrayOption : public Option
{
   public:

      DoubleArrayOption( const char* name, const char* defaultVal,
                        const char* usage, const char & cmd,
                        const int & N, double *const storage=NULL );
      DoubleArrayOption( const string& name, const string& defaultVal,
                        const string& usage, const char & cmd,
                        const int & N, double *const storage=NULL );
      DoubleArrayOption( const DoubleArrayOption &other );

      ~DoubleArrayOption() throw();

      virtual string getTypeName() const
      {
         return "DOUBLEARRAY";
      }

      void setValue( const string& s ) throw (UException);

      virtual double* getDoubleArray() const;

      virtual string toString() const;

   protected:
      const int N_;
      double * const storage_;
      double * const value_;
};



//! Hold an array of N ints.
class IntArrayOption : public Option
{
   public:

      IntArrayOption( const char* name, const char* defaultVal,
                      const char* usage, const char & cmd,
                      const int & N, int *const storage=NULL );
      IntArrayOption( const string& name, const string& defaultVal,
                      const string& usage, const char & cmd,
                      const int & N, int *const storage=NULL );
      IntArrayOption( const IntArrayOption &other );

      ~IntArrayOption() throw();

      virtual string getTypeName() const
      {
         return "INTARRAY";
      }

      void setValue( const string& s ) throw (UException);

      virtual int* getIntArray() const;

      virtual string toString() const;

   protected:
      const int N_;
      int * const storage_;
      int * const value_;
};

// hold an array of N strings
class StringArrayOption : public Option
{
public:
   
   StringArrayOption( const char* name, const char* defaultVal,
                      const char* usage, const char & cmd,
                      const int & N, string* const storage=NULL );
   StringArrayOption( const string& name, const string& defaultVal,
                      const string& usage, const char & cmd,
                      const int & N, string* const storage=NULL );
   StringArrayOption( const StringArrayOption &other );
   
   ~StringArrayOption() throw();
   
   virtual string getTypeName() const
   {
      return "STRINGARRAY";
   }
   
   void setValue( const string& s ) throw (UException);
   
   virtual string* getStringArray() const;
   
   virtual string toString() const;
   
protected:
   const int N_;
   string* const storage_;
   string* const value_;
};


//! Hold a N-dim \c Region.
class RegionArrayOption : public Option
{
   public:

      RegionArrayOption( const char* name, const char* defaultVal,
                         const char* usage, const char & cmd,
                         const int & N, Region * const = NULL );

      RegionArrayOption( const string& name, const string& defaultVal,
                         const string& usage, const char & cmd,
                         const int & N, Region * const = NULL );

      ~RegionArrayOption() throw()
      {}

      virtual string getTypeName() const
      {
         return "REGIONARRAY";
      }

      void setValue( const string& s ) throw (UException);

      virtual Region getRegionArray() const;

      virtual string toString() const;

      string toRegionString() const;

   protected:
      const int   N_;
      Region * storage_;
      Region value_;
};
//@}

//@}

}

#endif
