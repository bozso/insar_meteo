/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_data.cpp 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Claus A. Goessl <cag@usm.uni-muenchen.de>
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

#include<ltl/io/fits_data.h>

using util::Region;
using util::UException;

namespace ltl {

// return a subarray of type T of data segment
uint8_t* getRawRegionArray(uint8_t* const sourceptr, const int bitpix,
                           const Region& fullframe, const Region& region)
{
   // calculate new data length in bytes
   const size_t bytpix = size_t(abs(bitpix / 8));
   const size_t length = bytpix * region.getLength();
   // allocate data array
   uint8_t* const regionptr = new uint8_t [length];
   copyRawRegionArray(regionptr, sourceptr, bitpix, fullframe, region);
   return regionptr;
}

void copyRawRegionArray(uint8_t* const regionptr, uint8_t* const sourceptr,
                        const int bitpix,
                        const Region& fullframe, const Region& region)
{
   const size_t bytpix = size_t(abs(bitpix / 8));
   const size_t naxis = fullframe.getDim();
   size_t* const counterarray = new size_t[naxis];
   size_t* const regioncounter = counterarray - 1;
   for(size_t naxiscounter = 1; naxiscounter <= naxis; ++naxiscounter)
      regioncounter[naxiscounter] = 0;

   // calculate new data length in bytes
   const size_t length = bytpix * region.getLength();
   // allocate data array
   uint8_t* const destend = regionptr + length;
   uint8_t* destptr = regionptr;
   const size_t bytlinlength = bytpix * region.getLength(1);
   //cout<<"bytpix: "<<bytpix<<", length "<<length<<", naxis "<<naxis<<"\n";
   while(destptr < destend)
   { // copy until all done
      size_t srcoffset=0;
      for(size_t naxiscounter = naxis; naxiscounter > 1;)
      {
         srcoffset += region.getStart(naxiscounter) + regioncounter[naxiscounter] - 1;
         srcoffset *= fullframe.getLength(--naxiscounter);
      }
      srcoffset += region.getStart(1) + regioncounter[1] - 1;
      srcoffset *= bytpix;
      
      destptr = copyRawData(destptr, sourceptr + srcoffset, bytlinlength);

      size_t naxiscounter = 2;
      while(naxiscounter <= naxis)
      {
         if( (++regioncounter[naxiscounter]) < region.getLength(naxiscounter) )
            break;
         else
            regioncounter[naxiscounter]=0;
            ++naxiscounter;
      }
   }
   delete [] counterarray;
}


}
