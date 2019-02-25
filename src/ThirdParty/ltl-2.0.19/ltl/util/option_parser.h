/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: option_parser.h 552 2015-02-10 21:00:11Z rbryant $
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


#ifndef __UTIL_OPTION_PARSER__
#define __UTIL_OPTION_PARSER__

#include <ltl/config.h>

#include <list>
#include <map>
#include <string>
#include <iostream>
#include <fstream>

#include <cstring>

#include <ltl/util/u_exception.h>
#include <ltl/util/option.h>

using std::string;
using std::ostream;
using std::ofstream;
using std::list;
using std::map;
using std::endl;

namespace util {

/*! \addtogroup util_options
 *
 *  The option parsing library consists of a set of classes derived from \c util::Option
 *  which describe the types of the options the program accepts, for example a string,
 *  a float, an array of floats, etc.
 *
 *  There are two option readers, one constructing a list of options from the command line,
 *  the \c util::CommandLineReader, and one constructing a list of options from a file,
 *  the \c util::ConfigFileReader. Both are derived from the generic \c util::OptionReader.
 *  They can also be used in combination, i.e. reading options from a file and allowing
 *  the user to override values via the command line. The example code below should make
 *  things clear.
 *
 *  When using the option parsing library, the first thing to do is to tell it what options
 *  a program accepts. This is done by adding \c Option objects to an instance of \c OptionParser.
 *  Each option will have a data type, and will hold a long option name (used on the command line
 *  with --long-option [value], an optional short option form (e.g. -l), a default value, and,
 *  optionally, a pointer to the data type where the option will store its value (a pointer to,
 *  e.g., a global variable, or a member of a struct.
 *
 *  Next, a program would tell the \c OptionParser to parse the options it was given on the command
 *  line or in the config file. The \c OptionParser uses an \c OptionReader to do the actual reading.
 *  The values can then be probed using \c OptionParser::getOption().
 *
 *  Any arguments passed on the command line without a preceding '-' or '--' *AFTER* the last option
 *  are considered "file name arguments" and can be retrieved from the \c CommandLineReader
 *  using \c CommandLineReader::fileArguments().
 *
 *  The command line syntax is as follows: --long-name or -s, arguments separated by a space from
 *  the option, array arguments separated by commas without space. Bool flags toggle their default
 *  value.
 *
 *  \verbatim
   program --find-things --this-number 2.0 --array 1,2,3 -v 5.0 -s hello file1 file2 file3
   \endverbatim
 *
 *  In a config file, the syntax is long-option-name = value. '#' is the comment character.
 *  Value syntax the same as for the command line.
 *
 *  \verbatim
  find-things  = TRUE   # true|false|TRUE|FALSE for bool options
  this-number  = 2.0    # a float option
  array        = 1,2,3  # an integer array
  value        = 5.0    # another float, -v is short form, in the file we give the long name
  string       = hello  # -s was the short form
  \endverbatim
 *
 *  This config file can then be used together with the command line like this:
 *
 *  \verbatim
   program --config-file program.conf --find-things --array 5,6,7 file1 file2 file3
   \endverbatim
 * So that the program will find \c --find-things to be FALSE and array to be 5,6,7
 * (both overridden on the command line) and the remaining options as given in the config file.
 *
 *  The \c OptionParser::printUsage() method generates properly formatted and annotated usage
 *  information:
 *  \verbatim
  -h, --help <FLAG>                    [FALSE]
      Print usage information.

  -e, --expr <STRING>                  []
      The expression to evaluate. Mandatory. Function and variable
      names are case insensitive. The file operands are assigned to
      the variables in the expression in order.

  -o, --outfile <STRING>               [test.fits]
      Output filename.
  ...
 \endverbatim
 *
 *  To make everything clear, here is an example using just the command line.
 *
 *  \code
 *  // globals
 *  bool help;
 *  string outfile;
 *  int    size[3];
 *  CommandLineReader comline(argc, argv); // read from command line
 *  OptionParser flags(&comline);
 *  //                  type       --name    default  description             -short   ptr to value (optional)
 *  flags.addOption(new BoolOption("help", "FALSE", "Print usage information.", 'h', &help) );
 *  flags.addOption(new StringOption("outfile", "test.fits", "Output filename.", 'o', &outfile) );
 *  flags.addOption(new IntArrayOption("size", "0,0,0", "Size array", 's', 3, &size[0]) );
 *
 *  flags.parseOptions();
 *  list<string> files = comline.fileArguments(); // get the filenames
 *  \endcode
 *
 *  Finally, here is a more complex example using first a config file and then allowing options to be
 *  overridden by the command line.
 *
 *  \code
 *  // globals
 *  bool help;
 *  string outfile;
 *  int    size[3];
 *
 *  string cf = "";
 *  // config file name should be the first command line argument:
 *  if( argc >= 3 && (!strcmp(argv[1],"--config-file") || !strcmp(argv[1],"-c")) )
 *     cf = argv[2];
 *
 *  try
 *  {
 *    ConfigFileReader cfgfile( cf );
 *    OptionParser flags( &cfgfile );    // here we use the ConfigFileReader
 *
 *    //                  type       --name    default  description             -short   ptr to value (optional)
 *    flags.addOption(new BoolOption("help", "FALSE", "Print usage information.", 'h', &help) );
 *    flags.addOption(new StringOption("outfile", "test.fits", "Output filename.", 'o', &outfile) );
 *    flags.addOption(new IntArrayOption("size", "0,0,0", "Size array", 's', 3, &size[0]) );
 *
 *    // read the config file
 *    flags.parseOptions();
 *
 *    // then try again from the command line
 *    CommandLineReader comline(argc, argv); // read from command line
 *    flags.changeReader( &comline );
 *    flags.parseOptions();
 *  }
 *  catch (exception& e)
 *  {
 *    flags.printUsage(cerr);
 *    throw;
 *  }
 *  // finally, retrieve the file arguments
 *  list<string> files = comline.fileArguments(); // get the filenames
 *  \endcode
 *
 */

//@{

class OptionReader;

//! Generic parser object for options.
/*! Uses an object of type util::OptionReader
  to retrieve the options from whereever (config file, command-line, etc.)
*/
class OptionParser
{
   protected:
      friend ostream& operator<<( ostream& os, const OptionParser& op );
      typedef map<string,Option*> omap;
      typedef map<int,Option*>    onmap;
      
      OptionReader *reader_; //!< the \c OptionReader object we will use to read-in the options, either a \c CommandLineReader or a \c ConfigFileReader

      omap  options_;        //!< holds all options we know of indexed by name
      omap  cmd_options_;    //!< index of options by command line character
      onmap n_options_;      //!< holds options indexed by order of definition
      int   nopts_;          //!< number of options
   public:
      //! construct using an \c OptionReader, i.e. either a \c CommandLineReader or a \c ConfigFileReader.
      OptionParser( OptionReader* reader );
      virtual ~OptionParser();

      //! copy ctor.
      OptionParser( const OptionParser& other );

      //! delete all the \c Option objects we are holding.
      void deleteOptions( void );

      /*! switch the \c OptionReader to a different one, used when first reading a config
       *  file and then allowing to override option on the command line.
       */
      void changeReader( OptionReader* reader );

      //! add an option to our list of accepted options.
      void addOption( Option* option );

      //! retrieve an option by its name.
      Option* getOption( const string& name );

	  //! retrieve a list of the option names.
      const list<string> getOptionNames( ) const;

      //! read the options, parse them, and set their values.
      void parseOptions() throw( UException );

      /*! pretty-print the options, their descriptions, and default values in a format similar
      *   to man pages. automatic formatting and line-breaking. very nice!
      */
      void printUsage( ostream& os ) const;

      /*! write out the options in a format compatible with the \c ConfigFileReader.
      *   Convenient way to create a default config file.
      *   \c order_by_n specifies the order the options were added, otherwise alphabetic by name.
      */
      void writeConfig( const string& filename, 
                        const bool withComment = true, const bool order_by_n=true ) const throw( UException );

      //! get a string representation.
      string toString( const bool withComment = false ) const throw( UException );
};

/*! \relates util::OptionParser

  Write options to \e op. Will pretty-print the options, their default values, and their given values
  to the stream:
\verbatim
  expr             | e |                            |
  help             | h |                      FALSE | FALSE
  outfile          | o |                  test.fits | test.fits
  propagate-errors | p |                      FALSE | FALSE
  size             | s |                      0,0,0 | 0,0,0
  verbose          | v |                      FALSE | FALSE
  \endverbatim
*/
ostream& operator<<( ostream& os, const OptionParser& op );

//! An \b abstract base class for sources feeding into util::OptionParser .
/*! It defines the following interface
  supplying exactly three methods, namely
  \li string nextOptionName(),
  \li string nextOptionValue( const Option* op ),
  \li bool   done().
  These are then converted into objects of type Option.
*/
class OptionReader
{
   public:
      OptionReader()
      { }

      virtual ~OptionReader()
      { }

      virtual string nextOptionName()  = 0;
      virtual string nextOptionValue( const Option* op ) = 0;
      virtual bool   done() = 0;
};

//@}

}

#endif
