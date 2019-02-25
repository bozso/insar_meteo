/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: wcs.cpp 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#include <ltl/wcs.h>

namespace ltl {

//! simple parser ... has to be expanded
/*!
  This first version of a CCS parser assumes a very simple header.
  No more than a transformation from pixel x, y to RA, DEC for now.
  A more complex parser will follow when needs arise.
  the basic WCS template and the CCS class already could handle
  the complicated cases.
*/
FitsCCS::FitsCCS(const FitsHeader& hdr)
{
   int wcsaxes;
   try
   {
      wcsaxes = hdr.getInt("WCSAXES ");
   }
   catch(FitsException e){};
   
   {
      wcsaxes = hdr.getNaxis();
   }
   if(wcsaxes > 2)
      throw FitsException("Until now only simple 2d celestial WCS are supported.");
   FVector<double, 2> crpix(0.0);
   try
   {
      crpix(1) = hdr.getFloat("CRPIX1  ");
      crpix(2) = hdr.getFloat("CRPIX2  ");
   }
   catch(FitsException e){};

   // search for CDi_j and PCi_j
   FMatrix<double, 2, 2> pc(0.0);
   FVector<double, 2> cdelt(1.0);
   try
   {
      try
      {
         pc(1, 1) = hdr.getFloat("CD1_1   ");
      }
      catch(FitsException e){};
      try
      {
         pc(1, 2) = hdr.getFloat("CD1_2   ");
      }
      catch(FitsException e){};
      try
      {
         pc(2, 1) = hdr.getFloat("CD2_1   ");
      }
      catch(FitsException e){};
      try
      {
         pc(2, 2) = hdr.getFloat("CD2_2   ");
      }
      catch(FitsException e){};
      if( (pc(1,1) == pc(1,2)) && 
          (pc(2,1) == pc(2,2)) &&
          (pc(1,1) == pc(2,2)) &&
          (pc(1,1) == 0.0) )
      {
         try
         {
            pc(1, 1) = hdr.getFloat("PC1_1   ");
         }
         catch(FitsException e)
         { pc(1, 1) = 1.0; };
         try
         {
            pc(1, 2) = hdr.getFloat("PC1_2   ");
         }
         catch(FitsException e){};
         try
         {
            pc(2, 1) = hdr.getFloat("PC2_1   ");
         }
         catch(FitsException e){};
         try
         {
            pc(2, 2) = hdr.getFloat("PC2_2   ");
         }
         catch(FitsException e)
         { pc(2, 2) = 1.0; };
         try
         {      
            cdelt(1) = hdr.getFloat("CDELT1  ");
         }
         catch(FitsException e){};
         try
         {
            cdelt(2) = hdr.getFloat("CDELT2  ");
         }
         catch(FitsException e){};
      }
   }
   catch(FitsException e){};

   FVector<double, 2> crval(0.0);
   try
   {
      crval(1) = hdr.getFloat("CRVAL1  ");
      crval(2) = hdr.getFloat("CRVAL2  ");
   }
   catch(FitsException e){};

   string ctype = "";
   const string ctype_tan = string("-TAN");
   const string ctype_ra = string("RA--");
   const string ctype_dec = string("DEC-");
   try
   {
      string ctype1 = hdr.getString("CTYPE1  ");
      string ctype2 = hdr.getString("CTYPE2  ");
//       cerr << "'" << ctype1 << "'"
//            << "'" << ctype2 << "'" << endl;
//       cerr << "'" << ctype1.substr(4, 4) << "'"
//            << "'" << ctype2.substr(4, 4) << "'" << endl;
//       cerr << "'" << ctype1.substr(0, 4) << "'"
//            << "'" << ctype2.substr(0, 4) << "'" << endl;
      if( (ctype1.substr(4, 4) == ctype_tan) &&
          (ctype2.substr(4, 4) == ctype_tan) )
         ctype = ctype_tan;
      if( (ctype1.substr(0, 4) != ctype_ra) ||
          (ctype2.substr(0, 4) != ctype_dec) )
         cerr << "no RA / DEC CTYPE!" << endl;
   }
   catch(FitsException e){
      ctype = ctype_tan;
   };
   
   if(ctype == ctype_tan)
      ccs = new CCS_TAN(crpix, pc, cdelt, crval);
   else
      throw FitsException("Only TAN projection is supported. Expand parser!");
   
}

FVector<double, 2> FitsCCS::getRADEC(const FVector<double, 2>& p) const
{
//    cerr.precision(16);
//    cerr << "p: " << p << endl;
   return ccs->solve(p);
}

FVector<double, 2> FitsCCS::radecToPixel(const FVector<double, 2>& cc) const
{
//    cerr.precision(16);
//    cerr << "CC: " << cc << endl;
   return ccs->solve_inv(cc);
}


}
