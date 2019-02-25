/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: command_line_reader.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __UTIL_CLREADER__
#define __UTIL_CLREADER__

#include <ltl/config.h>

#include <list>
#include <vector>

#include <cstring>
#include <cctype>

#include <ltl/util/u_exception.h>
#include <ltl/util/option_parser.h>

using std::string;
using std::list;
using std::vector;

namespace util {

/*! \addtogroup util_options
*/
//@{

//! Implementation of an util::OptionReader for command line processing.
/*! 
  Also provides a list (or vector) of file arguments.

  Assumed syntax is:
  \verbatim
  command -s SHORTOPTION --long LONGOPTION ... file_arguments
  something -s string -f 1.234 --integerarray 1,2,3,4 file1 file2 file3
  \endverbatim
  File options must follow directly after command.
  No intervening file arguments!
*/
class CommandLineReader : public OptionReader
{
   public:
      CommandLineReader( int argc, char *argv[] );
      virtual ~CommandLineReader();

      virtual string nextOptionName() throw(UException);
      virtual string nextOptionValue( const Option* op ) throw(UException);
      virtual bool   done();

      list<string>   fileArguments();
      vector<string> fileArgumentsVector();
      string progName();

   protected:
      int   argc_;
      char** argv_;

      int   cur_;
      bool  done_;
};

//@}

}

#endif
