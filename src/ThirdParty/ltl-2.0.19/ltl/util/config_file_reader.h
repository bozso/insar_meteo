/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: config_file_reader.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __UTIL_CFREADER__
#define __UTIL_CFREADER__

#include <ltl/config.h>

#include <string>
#include <list>

#include <cctype>

#include <ltl/util/u_exception.h>
#include <ltl/util/option_parser.h>

using std::string;
using std::list;
using std::ifstream;

namespace util {

/*! \addtogroup util_options
*/
//@{


//! Implementation of an OptionReader for config file processing.
/*! 
  Syntax of config-file must be \n
  <tt>option = value</tt> \n
  Value must be of proper type.
  After '#' (comment indicator) rest of line will be ignored.
*/
class ConfigFileReader : public OptionReader
{
   public:
      ConfigFileReader( const string& filename );
      virtual ~ConfigFileReader();

      virtual string nextOptionName() throw(UException);
      virtual string nextOptionValue( const Option* op ) throw(UException);
      virtual bool   done();

      string fileName() const;

   protected:
      const string filename_;
      bool  done_;

      ifstream conffile_;
      string optionname_;
      string optionvalue_;

      bool nextLine();
};

//@}

}

#endif
