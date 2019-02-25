/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: command_line_reader.cpp 491 2011-09-02 19:36:39Z drory $
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

#include <ltl/util/command_line_reader.h>

namespace util {

// Implementation of an OptionReader for command line processing.
//
//

CommandLineReader::CommandLineReader( int argc, char *argv[] )
      : argc_(argc), argv_(argv), cur_(1), done_(argc<=1)
{
   // if first argumnet does not start with '-', it's a filename and
   // we're already done
   if( argc > 1 && *argv[1]!='-' )
      done_ = true;
}

CommandLineReader::~CommandLineReader()
{
}

string CommandLineReader::nextOptionName() throw(UException)
{
   char *s = argv_[cur_];
   string name;

   if( cur_ > argc_-1 )
      throw( UException( string("No more arguments left!") ) );

   if( strlen(s) < 2 )
      throw( UException(
                string("Token '")+s+string("' not an argument specifier!") ) );

   if( s[0] == '-' )
   {
      // we have an option
      if( s[1] == '-' )
      {
         // long option
         if( strlen(s+2) < 1 )
            throw( UException(
                      string("Token '")+s+string("' not an argument specifier!") ) );
         name = s+2;
         ++cur_;
      }
      else
      {
         // flag
         if( isspace(s[2]) )
            throw( UException(
                      string("Token '")+s+string("' not an argument specifier!") ) );
         name = s+1;
         ++cur_;
      }
   }
   else
   {
      // not starting with '-' : this means we are at the end of
      // the options, now come file arguments ....
      throw( UException(
                string("Token '")+s+string("' not an argument specifier!") ) );
   }
   return name;
}



string CommandLineReader::nextOptionValue( const Option* op ) throw(UException)
{
   char *s = argv_[cur_];
   string name;

   if( cur_ > argc_-1 && op->needsValue() )
      throw( UException( string("No more argument values left!") ) );

   if( op->needsValue() )
   {
      if( s[0] == '-' && s[1] != '.' && !isdigit(s[1]) )
         // starts with '-' but not a number ...
         throw( UException( string("Token '")+s+string("' does not seem like a valid value!") ) );
      ++cur_;
   }

   if( cur_ > argc_-1 || argv_[cur_][0] != '-' )
      done_=true;

   if( op->needsValue() )
   {
      return string(s);
   }
   else
   {
      return string( "toggle" );
   }
}



bool CommandLineReader::done()
{
   return cur_ > argc_-1 || done_;
}


string CommandLineReader::progName()
{
   string fullname = argv_[0];

   string::size_type lastslash = fullname.rfind( '/' ) + 1;
   return fullname.substr( lastslash, fullname.size() - lastslash );
}


list<string> CommandLineReader::fileArguments()
{
   list<string> l;
   while( cur_ < argc_ )
      l.push_back( string(argv_[cur_++]) );

   return l;
}

vector<string> CommandLineReader::fileArgumentsVector()
{
   vector<string> l(argc_ - cur_);
   for(int i = 0; cur_ < argc_; ++i, ++cur_)
      l[i] = argv_[cur_];
   return l;
}

}
