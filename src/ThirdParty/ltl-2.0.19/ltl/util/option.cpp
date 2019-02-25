/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: option.cpp 544 2014-08-01 14:48:03Z rbryant $
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

#include <ltl/util/option.h>

namespace util {

// Implementation for class Option
//
Option::Option( const char* name, const char* def,
                const char* usage, const char cmd )
      : name_(name), default_(def), usage_(usage), cmdlinechar_(cmd)
{ }

Option::Option( const string& name, const string& def,
                const string& usage, const char cmd )
      : name_(name), default_(def), usage_(usage), cmdlinechar_(cmd)
{ }

Option::~Option() throw()
{ }

const string& Option::getName() const
{
   return name_;
}

const string& Option::getDefault() const
{
   return default_;
}

const string& Option::getUsage() const
{
   return usage_;
}

char Option::getCmdLineChar() const
{
   return cmdlinechar_;
}


ostream& operator<<( ostream& os, Option& op )
{
   char buf[127];
   snprintf( buf, sizeof(buf), "%-16s | %c | %26s | ",
             op.getName().c_str(), (op.getCmdLineChar()==0 ? ' ' : op.getCmdLineChar()),
             op.getDefault().c_str() );
   os << buf << op.toString();

   return os;
}


// Implementation for class IntOption
//
IntOption::IntOption( const char* name, const char* def,
                      const char* usage, const char cmd,int *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

IntOption::IntOption( const string& name, const string& def,
                      const string& usage, const char cmd, int *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}


void IntOption::setValue( const string& s ) throw (UException)
{
   value_ = atoi( s.c_str() );
   if( storage_ )
      *storage_ = value_;
}

int IntOption::getInt() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}

string IntOption::toString() const
{
   char buf[63];
   snprintf( buf, sizeof(buf), "%d", getInt() );
   return string(buf);
}


// Implementation for class FloatOption
//
FloatOption::FloatOption( const char* name, const char* def,
                          const char* usage, const char cmd,
                          float *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

FloatOption::FloatOption( const string& name, const string& def,
                          const string& usage, const char cmd,
                          float *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

void FloatOption::setValue( const string& s ) throw (UException)
{
   value_ = atof( s.c_str() );
   if( storage_ )
      *storage_ = value_;
}

float FloatOption::getFloat() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}

string FloatOption::toString() const
{
   char buf[63];
   snprintf( buf, sizeof(buf), "%g", getFloat() );
   return string(buf);
}


// Implementation for class DoubleOption
//
DoubleOption::DoubleOption( const char* name, const char* def,
                            const char* usage, const char cmd,
                            double *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

void DoubleOption::setValue( const string& s ) throw (UException)
{
   value_ = atof( s.c_str() );
   if( storage_ )
      *storage_ = value_;
}

float DoubleOption::getDouble() const
{
   return value_;
}

string DoubleOption::toString() const
{
   char buf[63];
   snprintf( buf, sizeof(buf), "%g", getDouble() );
   return string(buf);
}


// Implementation for class BoolOption
//
BoolOption::BoolOption( const char* name, const char* def,
                        const char* usage, const char cmd,
                        bool *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

BoolOption::BoolOption( const string& name, const string& def,
                        const string& usage, const char cmd,
                        bool *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

void BoolOption::setValue( const string& s ) throw (UException)
{
   if( s == "toggle" ) value_ = !value_;
   else
      value_ = ( s!="FALSE" && s!="false" && s!="NO" && s!="no" &&
                 s!="F" && s!="f" && s!="N" && s!="n" && s!="0" );
   if( storage_ )
      *storage_ = value_;
}

bool BoolOption::getBool() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}

string BoolOption::toString() const
{
   char buf[7];
   snprintf( buf, sizeof(buf), "%s", (getBool()==true ? "TRUE" : "FALSE") );
   return string(buf);
}


// Implementation for class StringOption
//
StringOption::StringOption( const char* name, const char* def,
                            const char* usage, const char cmd,
                            string *const storage)
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

StringOption::StringOption( const string& name, const string& def,
                            const string& usage, const char cmd,
                            string *const storage )
      : Option( name, def, usage, cmd ), storage_(storage)
{
   setValue( default_ );
}

void StringOption::setValue( const string& s ) throw (UException)
{
   value_ = s;
   if( storage_ )
      *storage_ = value_;
}

string StringOption::getString() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}

string StringOption::toString() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}


// Implementation for class FloatArrayOption
//
FloatArrayOption::FloatArrayOption( const char* name, const char* def,
                                    const char* usage, const char & cmd,
                                    const int & N, float *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new float[N_])
{
   setValue( default_ );
}

FloatArrayOption::FloatArrayOption( const string& name, const string& def,
                                    const string& usage, const char & cmd,
                                    const int & N, float *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new float[N_])
{
   setValue( default_ );
}

FloatArrayOption::FloatArrayOption( const FloatArrayOption & other )
      : Option( other.name_, other.default_, other.usage_, other.cmdlinechar_ ),
      N_(other.N_), storage_(other.storage_), value_(new float[N_])
{
   for(int i = 0; i < N_; ++i)
      value_[i] = other.value_[i];
}

FloatArrayOption::~FloatArrayOption() throw()
{
   delete [] value_;
}

void FloatArrayOption::setValue( const string& s ) throw (UException)
{
   int i=0;
   string str = s, token;
   string::size_type end = 1;

   // comma separated string of floats
   while( end != str.npos )
   {
      if( i>=N_ )
         throw(UException(string("Too many elements in FloatArrayOption "+name_+"!")));
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      value_[i] = atof( token.c_str() );
      if( storage_ )
         storage_[i] = value_[i];
      ++i;
   }
}

float* FloatArrayOption::getFloatArray() const
{
   if( storage_ )
      return storage_;
   else
      return value_;
}

string FloatArrayOption::toString() const
{
   float* tmp;
   if( storage_ )
      tmp = storage_;
   else
      tmp = value_;

   string s = "";
   int i;
   char buf[64];
   for( i=0; i < N_-1; ++i )
   {
      snprintf( buf, sizeof(buf), "%g,", tmp[i] );
      s += buf;
   }
   snprintf( buf, sizeof(buf), "%g", tmp[i] );
   s += buf;

   return s;
}

// Implementation for class DoubleArrayOption
//
DoubleArrayOption::DoubleArrayOption( const char* name, const char* def,
                                    const char* usage, const char & cmd,
                                    const int & N, double *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new double[N_])
{
   setValue( default_ );
}

DoubleArrayOption::DoubleArrayOption( const string& name, const string& def,
                                    const string& usage, const char & cmd,
                                    const int & N, double *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new double[N_])
{
   setValue( default_ );
}

DoubleArrayOption::DoubleArrayOption( const DoubleArrayOption & other )
      : Option( other.name_, other.default_, other.usage_, other.cmdlinechar_ ),
      N_(other.N_), storage_(other.storage_), value_(new double[N_])
{
   for(int i = 0; i < N_; ++i)
      value_[i] = other.value_[i];
}

DoubleArrayOption::~DoubleArrayOption() throw()
{
   delete [] value_;
}

void DoubleArrayOption::setValue( const string& s ) throw (UException)
{
   int i=0;
   string str = s, token;
   string::size_type end = 1;

   // comma separated string of doubles
   while( end != str.npos )
   {
      if( i>=N_ )
         throw(UException(string("Too many elements in DoubleArrayOption "+name_+"!")));
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      value_[i] = atof( token.c_str() );
      if( storage_ )
         storage_[i] = value_[i];
      ++i;
   }
}

double* DoubleArrayOption::getDoubleArray() const
{
   if( storage_ )
      return storage_;
   else
      return value_;
}

string DoubleArrayOption::toString() const
{
   double *tmp;
   if( storage_ )
      tmp = storage_;
   else
      tmp = value_;

   string s = "";
   int i;
   char buf[64];
   for( i=0; i < N_-1; ++i )
   {
      snprintf( buf, sizeof(buf), "%g,", tmp[i] );
      s += buf;
   }
   snprintf( buf, sizeof(buf), "%g", tmp[i] );
   s += buf;

   return s;
}

// Implementation for class IntArrayOption
//
IntArrayOption::IntArrayOption( const char* name, const char* def,
                                const char* usage, const char & cmd,
                                const int & N, int *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new int[N_])
{
   setValue( default_ );
}
IntArrayOption::IntArrayOption( const string& name, const string& def,
                                const string& usage, const char & cmd,
                                const int & N, int *const storage )
      : Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new int[N_])
{
   setValue( default_ );
}
IntArrayOption::IntArrayOption( const IntArrayOption & other )
      : Option( other.name_, other.default_, other.usage_, other.cmdlinechar_ ),
      N_(other.N_), storage_(other.storage_), value_(new int[N_])
{
   for(int i = 0; i < N_; ++i)
      value_[i] = other.value_[i];
}

IntArrayOption::~IntArrayOption() throw()
{
   delete [] value_;
}

void IntArrayOption::setValue( const string& s ) throw (UException)
{
   int i=0;
   string str = s, token;
   string::size_type end = 1;

   // comma separated string of integers
   while( end != str.npos )
   {
      if( i>=N_ )
         throw(UException(string("Too many elements in IntArrayOption "+name_+"!")));
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      value_[i] = atoi( token.c_str() );
      if( storage_ )
         storage_[i] = value_[i];
      ++i;
   }
}

int* IntArrayOption::getIntArray() const
{
   if( storage_ )
      return storage_;
   else
      return value_;
}

string IntArrayOption::toString() const
{
   int *tmp;
   if( storage_ )
      tmp = storage_;
   else
      tmp = value_;

   string s = "";
   int i;
   char buf[64];
   for( i = 0; i < N_-1; ++i )
   {
      snprintf( buf, sizeof(buf), "%d,", tmp[i] );
      s += buf;
   }
   snprintf( buf, sizeof(buf), "%d", tmp[i] );
   s += buf;

   return s;
}

// Implementation for class StringArrayOption
//
StringArrayOption::StringArrayOption( const char* name, const char* def,
                                      const char* usage, const char & cmd,
                                      const int & N, string *const storage )
: Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new string[N_])
{
   setValue( default_ );
}
StringArrayOption::StringArrayOption( const string& name, const string& def,
                                      const string& usage, const char & cmd,
                                      const int & N, string *const storage )
: Option( name, def, usage, cmd ), N_(N), storage_(storage), value_(new string[N_])
{
   setValue( default_ );
}
StringArrayOption::StringArrayOption( const StringArrayOption & other )
: Option( other.name_, other.default_, other.usage_, other.cmdlinechar_ ),
N_(other.N_), storage_(other.storage_), value_(new string[N_])
{
   for(int i = 0; i < N_; ++i)
      value_[i] = other.value_[i];
}

StringArrayOption::~StringArrayOption() throw()
{
   delete [] value_;
}

void StringArrayOption::setValue( const string& s ) throw (UException)
{
   int i=0;
   string str = s, token;
   string::size_type end = 1;

   // comma separated string of strings
   while( end != str.npos )
   {
      if( i>=N_ )
         throw(UException(string("Too many elements in StringArrayOption "+name_+"!")));
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() ); 
      }
      value_[i] = token;
      if( storage_ )
         storage_[i] = value_[i];
      ++i;
   }
}

string* StringArrayOption::getStringArray() const
{        
   if( storage_ )
      return storage_;
   else
      return value_;
}

string StringArrayOption::toString() const
{
   string* tmp;
   if( storage_ )
      tmp = storage_;
   else
      tmp = value_;

   string s = "";
   int i;
   for( i = 0; i < N_-1; ++i )
   {
      s += tmp[i];
      s += ",";
   }

   s += tmp[i];

   return s;
}


// Implementation for class RegionArrayOption
//

RegionArrayOption::RegionArrayOption( const char* name, const char* def,
                                      const char* usage, const char & cmd,
                                      const int & N,
                                      Region * const storage ) :
      Option(name, def, usage, cmd), N_(N), storage_(storage), value_(N_)
{
   setValue( default_ );
}
RegionArrayOption::RegionArrayOption( const string& name, const string& def,
                                      const string& usage, const char & cmd,
                                      const int & N,
                                      Region * const storage ) :
      Option(name, def, usage, cmd), N_(N), storage_(storage), value_(N_)
{
   setValue( default_ );
}

void RegionArrayOption::setValue( const string& s ) throw (UException)
{
   int i=1;
   string str = s, token;
   string::size_type end = 1;

   // comma separated string of dim_i_start:dim_i_end
   while( end != str.npos )
   {
      if( i > N_ )
         throw(UException(string("Too many elements in RegionArrayOption "+name_+"!")));
      end = str.find_first_of( ":", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      const int startvalue = atoi( token.c_str() );
      end = str.find_first_of( ",", 0 );
      if( end != str.npos )
      {
         token = str.substr( 0, end );
         str.erase( 0, end+1 );
      }
      else
      {
         token = str.substr( 0, str.length() );
         str.erase(0, str.length() );
      }
      const int endvalue = atoi( token.c_str() );
      value_.setRange(i,
                      ( (endvalue > startvalue) ? startvalue : endvalue ),
                      ( (endvalue > startvalue) ? endvalue : startvalue )); // swap if s>e
      if( storage_ )
         (*storage_).setRange(i, value_.getStart(i), value_.getEnd(i));
      ++i;
   }
}

Region RegionArrayOption::getRegionArray() const
{
   if( storage_ )
      return *storage_;
   else
      return value_;
}

string RegionArrayOption::toString() const
{
   const Region* tmp;
   if( storage_ )
      tmp = storage_;
   else
      tmp = &value_;

   string s = "";
   int i;
   char buf[63];
   for( i=1; i<N_; ++i )
   {
      sprintf( buf, "%d:%d,", tmp->getStart(i), tmp->getEnd(i) );
      s += buf;
   }
   sprintf( buf, "%d:%d", tmp->getStart(i), tmp->getEnd(i) );
   s += buf;

   return s;
}

string RegionArrayOption::toRegionString() const
{
   return string("[") + toString() + string("]");
}

}
