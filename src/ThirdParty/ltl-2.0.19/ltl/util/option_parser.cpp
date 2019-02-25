/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: option_parser.cpp 552 2015-02-10 21:00:11Z rbryant $
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


#include <ltl/util/option_parser.h>

namespace util {
using std::endl;

// Generic parser object for options. Uses an object of type OptionReader
// to retrieve the options from whereever (config file, command-line, etc.)
// OptionReader has to supply exactly three methods, namely
//    string nextOptionName(),
//    string nextOptionValue(),
//    bool   done()
// These are then converted into objects of type Option.
//

OptionParser::OptionParser( OptionReader* reader )
   : reader_( reader ), nopts_(0)
{ }

OptionParser::OptionParser( const OptionParser& other )
   : reader_(other.reader_), options_(other.options_), 
     cmd_options_(other.cmd_options_),
     n_options_(other.n_options_), nopts_(other.nopts_)
{ }


OptionParser::~OptionParser()
{
}


void OptionParser::deleteOptions( void )
{
   omap::const_iterator i;
   for( i=options_.begin(); i!=options_.end(); ++i )
      delete (*i).second;
   options_.clear();
   cmd_options_.clear();
   n_options_.clear();
}

void OptionParser::changeReader( OptionReader* reader )
{
   reader_ = reader;
}

void OptionParser::addOption( Option* option )
{
   // add to option map indexed by name
   if (options_[option->getName()] != 0)
         throw(UException( string("An option named "+option->getName()+" already exists!")));
   options_[option->getName()] = option;

   // add to index by cmd line char
   char c[2];
   c[1] = 0;
   c[0] = option->getCmdLineChar();
   if( c[0] )
   {
      string opt_c = c;
      if (cmd_options_[opt_c] != 0){
         throw(UException( string("Command line char "+opt_c+" is already in use!")));
      }
      cmd_options_[opt_c] = option;
   }
   // add to index by number
   n_options_[++nopts_] = option;
}


Option* OptionParser::getOption( const string& name )
{
   return options_[name];
}

const list<string> OptionParser::getOptionNames( ) const
{
   list<string> reply;
   for (omap::const_iterator itr = options_.begin(); itr != options_.end(); ++itr)
	   reply.push_back(((*itr).second)->getName());
   return reply;
}

void OptionParser::parseOptions() throw( UException )
{
   while( !reader_->done() )
   {
      // get the next option name
      string o = reader_->nextOptionName();

      // see if we can find the option ...
      omap::iterator i;
      if( o.length() > 1 )
      {
         // long option
         i = options_.find( o );
         if( i==options_.end() )
            throw(UException(string("Unknown long option '"+o+"' encountered!")));
      }
      else
      {
         // short option
         i = cmd_options_.find( o );
         if( i==cmd_options_.end() )
            throw(UException(string("Unknown short option '"+o+"' encountered!")));
      }

      // if the option is not a flag, read it's value
      Option* op = (*i).second;
      op->setValue( reader_->nextOptionValue( op ) );
   }
}



void OptionParser::printUsage( ostream& os ) const
{
   onmap::const_iterator i;
   for( i=n_options_.begin(); i!=n_options_.end(); ++i )
   {
      Option &o = *((*i).second);

      // the option syntax part
      string s = "    ";
      char c = o.getCmdLineChar();
      if( c )
      {
         s += "-";
         s += c;
         s += ", ";
      }
      s += "--";
      s += o.getName();
      s += " <";
      s += o.getTypeName();
      s += ">";

      int pad = 40 - s.length();
      while ( --pad > 0 )
         s += " ";

      s += "  [";
      s += o.getDefault();
      s += "]";

      os << s << endl;

      // the help string
      string u = o.getUsage();
      char* str = const_cast<char*>(u.c_str());
      string indent = "        ";
      char* st = str;
      string out = "";
      int pos = indent.length();

      os << indent;

      st = strtok( st, " \t\n" );
      while( st )
      {
         if( pos + strlen( st ) < 72 )
         {
            os << st << " ";
            pos += strlen(st) + 1;
         }
         else
         {
            os << endl << indent << st << " ";
            pos = indent.length() + strlen(st) + 1;
         }

         st = strtok( NULL, " \t\n" );
      }
      os << endl << endl;
   }
}

void OptionParser::writeConfig( const string& filename,
                                const bool withComment, const bool order_by_n ) const
   throw( UException )
{
   ofstream outfile;
   outfile.open(filename.c_str());
   if(!outfile.is_open())
      throw (UException( string("Cannot open file ") +
                         filename + string(" for writing") ));


   if( order_by_n )
   {
      onmap::const_iterator i;
      for( i=n_options_.begin(); i!=n_options_.end(); ++i )
      {
         const Option &o = *((*i).second);
         outfile << o.getName() << "\t= " << o.toString();
         if(withComment)
            outfile << "\t# " << o.getUsage();
         outfile << endl;
      }
   }
   else
   {
      omap::const_iterator i;
      for( i=options_.begin(); i!=options_.end(); ++i )
      {
         const Option &o = *((*i).second);
         outfile << o.getName() << "\t= " << o.toString();
         if(withComment)
            outfile << "\t# " << o.getUsage();
         outfile << endl;
      }
   }
   outfile.close();
}

string OptionParser::toString( const bool withComment ) const
   throw( UException )
{
   string outstring = "";
   map<string, Option*>::const_iterator i = options_.begin();
   while( i != options_.end() )
   {
      const Option &o = *((*i).second);
      outstring += o.getName() + string("=") + o.toString();
      if(withComment)
         outstring += string(" # ") + o.getUsage();
      ++i;
      if( i!= options_.end())
         outstring += string("; ");
   }
   return outstring;
}


ostream& operator<<( ostream& os, const OptionParser& op )
{
   map<string,Option*>::const_iterator i;
   for( i=op.options_.begin(); i!=op.options_.end(); ++i )
      os << *((*i).second) << endl;

   return os;
}

}
