/* -*- C++ -*-
*
* ---------------------------------------------------------------------
* $Id: testconfreader.cpp 363 2008-07-10 15:09:44Z drory $
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

#define LTL_RANGE_CHECKING
//#define LTL_DEBUG_EXPRESSIONS

#define LTL_TEMPLATE_LOOP_LIMIT 0

#include <ltl/marray.h>
#include <ltl/util/config_file_reader.h>

#include <iostream>
#include <string>

using namespace util;
using namespace std;

int main(int argc, char **argv)
{
   cerr<<"Testing config file reading (step 1)..."<<endl;

   float admin_port, admin_port2;
   int relay_port, relay_port2;
   string relay_ip, relay_ip2;

   ConfigFileReader& cfr = *new ConfigFileReader( "ltl-test.conf" );
   OptionParser& op = *new OptionParser( &cfr );

   try {
      op.addOption( new StringOption( "RELAY_IP", "129.187.204.70",
                                      "IP address of tcs-relay",
                                      0 , &relay_ip) );
      op.addOption( new IntOption( "RELAY_COM_PORT", "7999", 
                                   "COM Port on tcs-relay",
                                   0 , &relay_port) );
      op.addOption( new FloatOption( "ADMIN_PORT", "5242.56", 
                                   "Port for admin commands",
                                   0 , &admin_port) );
      // parse command line ...
      op.parseOptions();

   } catch( UException e ) {
      cerr << "Caught exception : " << e.what() << endl;
      exit(1);
   }

   cerr<<"Testing config file writing (step 2)..."<<endl;

   op.writeConfig( "ltl-test.conf" );

   cerr<<"Testing config file reading (step 3)..."<<endl;

   ConfigFileReader& cfr2 = *new ConfigFileReader( "ltl-test.conf" );
   OptionParser& op2 = *new OptionParser( &cfr2 );

   try {
      op2.addOption( new StringOption( "RELAY_IP", "129.187.204.71",
                                      "IP address of tcs-relay",
                                      0 , &relay_ip2) );
      op2.addOption( new IntOption( "RELAY_COM_PORT", "7997", 
                                   "COM Port on tcs-relay",
                                   0 , &relay_port2) );
      op2.addOption( new FloatOption( "ADMIN_PORT", "5242.34", 
                                   "Port for admin commands",
                                   0 , &admin_port2) );
      // parse command line ...
      op2.parseOptions();

   } catch( UException e ) {
      cerr << "Caught exception : " << e.what() << endl;
      exit(1);
   }
   
   cerr<<"Comparing results (step 4)..."<<endl;

   LTL_ASSERT_( (admin_port == admin_port2),
                "Option read of float failed" );
   
   LTL_ASSERT_( (relay_port == relay_port2 ),
                "Option read of int failed" );
   
   LTL_ASSERT_( (relay_ip == relay_ip2 ),
                "Option read of string failed" );

   return 0;
}

