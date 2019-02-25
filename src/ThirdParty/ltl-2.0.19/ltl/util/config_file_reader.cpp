/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: config_file_reader.cpp 491 2011-09-02 19:36:39Z drory $
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


#include <ltl/util/config_file_reader.h>

namespace util {

// Implementation of an OptionReader for command line processing.

ConfigFileReader::ConfigFileReader( const string& filename )
      : filename_(filename), done_(false)
{
   conffile_.open(filename_.c_str());
   if(!conffile_.is_open())
      done_ = true;
   else
      done_ = !nextLine();
}

ConfigFileReader::~ConfigFileReader()
{
   if(conffile_.is_open())
      conffile_.close();
}

// true as long there is any valid next line
bool ConfigFileReader::nextLine()
{
   bool is_invalid = true;
   do // as long as there is no valid new line and not end of file
   {
      if( conffile_.eof() )
         return false; // break on end of file
      string line;
      getline(conffile_, line); // read line
      const string::size_type compos = line.find_first_of('#');
      if(compos != line.npos)
         line = line.substr(0, compos); // discard comment
      const string::size_type eqpos = line.find_first_of('=');
      if(eqpos != line.npos) // is there a '=' ?
      {
         // parse option name
         optionname_ = line.substr(0, eqpos);
         string::size_type start = 0;
         while( (start < optionname_.length()) && isspace(optionname_[start]) )
            ++start; // discard leading blanks
         if(start < optionname_.length()) // not everything blank
         {
            string::size_type end = start + 1;
            while( (end < optionname_.length()) && !isspace(optionname_[end]) )
               ++end;
            string::size_type check = end + 1;
            while( (check < optionname_.length()) && isspace(optionname_[check]) )
               ++check;
            if(check >= optionname_.length()) // optionname is ok?
            {
               optionname_ = optionname_.substr(start, end - start);
               // parse option value
               optionvalue_ = "";
               start = eqpos + 1;
               while( (start < line.length()) && isspace(line[start]) )
                  ++start; // skip leading blanks
               if(start < line.length())
               { // not everything blank before opt. comment
                  if(line[start] == '\"')
                  { // is it a "quoted" string
                     if((++start) < line.length())
                     {
                        end = line.find_first_of('\"', start);
                        if(end == line.npos)
                           end = line.length();
                     }
                     else
                        end = line.length();
                  }
                  else
                  {
                     end = start + 1;
                     while( (end < line.length()) && !isspace(line[end]) )
                        ++end;
                  }
                  optionvalue_ = line.substr(start, end-start);
               }
               is_invalid = false;
            }
         }
      }
   }
   while (is_invalid);
   return true;
}

string ConfigFileReader::nextOptionName() throw(UException)
{
   if( done_ )
      throw( UException( string("No more arguments left!") ) );
   return optionname_;
}

string ConfigFileReader::nextOptionValue( const Option* op ) throw(UException)
{
   if( done_ )
      throw( UException( string("No more argument values left!") ) );
   const string returnvalue = optionvalue_;
   done_ = !nextLine();
   return returnvalue;
}

string ConfigFileReader::fileName() const
{
   return filename_;
}

bool ConfigFileReader::done()
{
   return done_;
}

}
