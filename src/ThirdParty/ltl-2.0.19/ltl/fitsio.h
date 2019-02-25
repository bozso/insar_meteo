/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fitsio.h 553 2015-02-17 02:03:32Z cag $
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

/* \file fitsio.h
  I/O adapter class fitsfile \>\> ltl::MArray and fitsfile \<\< ltl::MArray.
*/

#ifndef __FITSIO_H__
#define __FITSIO_H__

#include <ltl/config.h>
#include <ltl/fits.h>
#include <ltl/marray.h>
#include <ltl/misc/exceptions.h>

#include <cstdlib> // for abs()

// templates to read from fitsfile class

namespace ltl {

// read data
// check dimensions
template<class T, int N>
MArray<T, N> getFitsMArray(FitsIn& fitsfile)
{
   const int dimension = fitsfile.getNaxis();
   int collapse = 0;
   const size_t length = fitsfile.getRegionLength();
   int dim_ [N];
   for(int i = 0; i<N; dim_[i++]=1);

   if(length)
   {
      for(int i = 0; (i<dimension) && ((i-collapse)<N); ++i)
      {
         dim_[i-collapse] = fitsfile.getRegionLength(i+1);
         if(dim_[i-collapse] == 1)
            ++collapse;
      }
      for(int i = dimension; i > N; --i )
         if(fitsfile.getRegionLength(i) == 1) ++collapse; // otherwise we would have missed all collapses > N
   }
   else
      for(int i = 0; (i<dimension) && (i<N); ++i)
         dim_[i] = fitsfile.getNaxis(i+1);

   if( N < dimension - collapse )
      throw FitsException("NAXIS or slice cannot match MArray dimensions");

   T* dataptr = fitsfile.getDataArray(T(0));

   return MArray<T, N>(dataptr, dim_);
}

template<class T, int N>
void insertFitsMArray( MArray<T, N>& a, FitsIn& fitsfile)
{
   if(N > fitsfile.getNaxis())
      throw FitsException("MArray cannot be filled with FITS file");

   util::Region fregion(fitsfile.getNaxis());
   // setup a initial region
   if(fitsfile.getRegionLength() == 0)
   {
      for(int i = 1; i <= N; ++i)
         fregion.setRange( i, 1,
                           (a.length(i) < fitsfile.getNaxis(i)) ?
                           a.length(i) : fitsfile.getNaxis(i) );
      for(int i = N+1; i <= fitsfile.getNaxis(); ++i)
         fregion.setRange( i, 1, 1);

      // test if a is exactly conformable with file
      bool exactmatch = (fitsfile.getNaxis() == N);
      for(int i = 1; exactmatch && (i <=N); ++i)
         exactmatch = exactmatch && (fregion.getLength(i) == (unsigned int)fitsfile.getNaxis(i));
      if(!exactmatch)
         fitsfile.setRegion(fregion);
      else
         fitsfile.resetRegion();
   }
	 else
		 fregion.setRegion(fitsfile.getRegion());

   // actually read data
   if(fitsfile.getRegionLength() == 0)
   {
      typename MArray<T, N>::iterator i = a.begin();
      fitsfile.readDataArray(i);
   }
   else{
      if( a.nelements() == fregion.getLength() )
      {
         typename MArray<T, N>::iterator i = a.begin();
         fitsfile.readRegionArray(i);
      }
      else
      {
         a = T(0);
         MArray<T, N> b(getMArrayRegion(a, fregion));
         typename MArray<T, N>::iterator i = b.begin();
         fitsfile.readRegionArray(i);
      }

			// increment region
      bool carriage_return = true;      
      int i = 1;
      while(carriage_return && (i <= N))
      {
         if(carriage_return)
         {
            const int actend = fitsfile.getRegionEnd(i);
            const int actnaxis = fitsfile.getNaxis(i);
            const int actalength = a.length(i);
            // end of line?
            if(actend == actnaxis) // yes => new line and carriage return
               fregion.setRange( i, 1,
                                 (actalength < actnaxis) ? 
                                 actalength : actnaxis );
            else // no => step ahead
            {
               const int newend = actend + actalength;
               if(newend < actnaxis)
                  fregion.setRange(i, actend + 1, newend);
               else
                  fregion.setRange(i, actend + 1, actnaxis);
               carriage_return = false;
            }
         }
         ++i;
      }
      while(carriage_return && (i <= fitsfile.getNaxis()))
      {
         if(carriage_return)
         {
            const int actend = fitsfile.getRegionEnd(i);
            const int actnaxis = fitsfile.getNaxis(i);
            // end of line?
            if(actend == actnaxis) // yes => new line and carriage return
               fregion.setRange( i, 1, 1);
            else // no => step ahead
            {
               const int newend = actend + 1;
               fregion.setRange(i, newend, newend);
               carriage_return = false;
            }
         }
         ++i;
      }

      if(carriage_return)
      {
         fitsfile.resetRegion();
         //throw FitsException("FITS file already completely read, next try starts over.");
      }
      fitsfile.setRegion(fregion);
   }
}

template<class T>
MArray<T, 1> getMArrayRegion(MArray<T, 1>& a, const util::Region& fregion)
{
   return a( Range(fregion.getStart(1), fregion.getLength(1)) );
}

template<class T>
MArray<T, 2> getMArrayRegion(MArray<T, 2>& a, const util::Region& fregion)
{
   return a( Range(fregion.getStart(1), fregion.getEnd(1)),
             Range(fregion.getStart(2), fregion.getEnd(2)));
}

template<class T>
MArray<T, 3> getMArrayRegion(MArray<T, 3>& a, const util::Region& fregion)
{
   return a( Range(fregion.getStart(1), fregion.getEnd(1)),
             Range(fregion.getStart(2), fregion.getEnd(2)),
             Range(fregion.getStart(3), fregion.getEnd(3)));
}

template<class T>
MArray<T, 4> getMArrayRegion(MArray<T, 4>& a, const util::Region& fregion)
{
   return a( Range(fregion.getStart(1), fregion.getEnd(1)),
             Range(fregion.getStart(2), fregion.getEnd(2)),
             Range(fregion.getStart(3), fregion.getEnd(3)),
             Range(fregion.getStart(4), fregion.getEnd(4)));
}

template<class T>
MArray<T, 5> getMArrayRegion(MArray<T, 5>& a, const util::Region& fregion)
{
   return a( Range(fregion.getStart(1), fregion.getEnd(1)),
             Range(fregion.getStart(2), fregion.getEnd(2)),
             Range(fregion.getStart(3), fregion.getEnd(3)),
             Range(fregion.getStart(4), fregion.getEnd(4)),
             Range(fregion.getStart(5), fregion.getEnd(5)));
}

/*! \addtogroup ma_fits_io
*/
//@{

/*! \relates ltl::FitsIn
  Read FITS data into ltl::MArray.
*/
template<class T, int N>
FitsIn& operator>>(FitsIn& fitsfile, MArray<T, N> & a)
{
   if(a.isAllocated())
      insertFitsMArray<T, N>(a, fitsfile);
   else
      a.makeReference(getFitsMArray<T, N>(fitsfile));
   return fitsfile;
}

//@}

/// \cond DOXYGEN_IGNORE
template<class T>
struct Bitpix
{
   enum { bitpix = 0 };
};

template<>
struct Bitpix<unsigned char>
{
   enum { bitpix = 8 };
};

template<>
struct Bitpix<char>
{
   enum { bitpix = 8 };
};

template<>
struct Bitpix<short int>
{
   enum { bitpix = 16 };
};

template<>
struct Bitpix<int>
{
   enum { bitpix = 32 };
};

template<>
struct Bitpix<float>
{
   enum { bitpix = -32 };
};

template<>
struct Bitpix<double>
{
   enum { bitpix = -64 };
};
/// \endcond

/*! \addtogroup ma_fits_io
*/
//@{

/*! \relates ltl::FitsOut
  Write ltl::MArray to FITS file.
*/
template<class T, int N>
FitsOut& operator<<(FitsOut& fitsfile, const MArray<T, N>& a)
{

   // get new NAXIS
   int naxis_i [N];
   for(int i = 0; i < N; ++i)
      naxis_i[i] = a.length(i+1);

   // if neccessary set default BITPIX
   if( fitsfile.getBitpixOut() == 0 )
      fitsfile.setBitpixOut(Bitpix<T>::bitpix);

   // if neccessary reset ORIGIN
   if( (fitsfile.getOrigin()).length() < 1 )
      fitsfile.setOrigin("LTL FITS IO class, September 2010");

   if(fitsfile.isRegion())
   {
      // write data
      typename MArray<T, N>::const_iterator i = a.begin();
      fitsfile.writeRegionArray(i);
   }
   else
   {
      // map file
      fitsfile.openData(fitsfile.getBitpixOut(), N, naxis_i);
      // write data
      typename MArray<T, N>::const_iterator i = a.begin();
      fitsfile.writeDataArray(i);
   }
   // close file
   fitsfile.closeData();
   return fitsfile;
}

/*! \relates ltl::FitsExtensionOut
  Write ltl::MArray to FITS file with extensions.
*/
template<class T, int N>
FitsExtensionOut& operator<<(FitsExtensionOut& fitsfile, const MArray<T, N>& a)
{

   // get new NAXIS
   int naxis_i [N];
   for(int i = 0; i < N; ++i)
      naxis_i[i] = a.length(i+1);

   // if neccessary set default BITPIX
   if( fitsfile.getBitpixOut() == 0 )
      fitsfile.setBitpixOut(Bitpix<T>::bitpix);

   // if neccessary reset ORIGIN
   if( (fitsfile.getOrigin()).length() < 1 )
      fitsfile.setOrigin("LTL FITS IO class, September 2010");

   if(fitsfile.isRegion())
   {
      // write data
      typename MArray<T, N>::const_iterator i = a.begin();
      fitsfile.writeRegionArray(i);
   }
   else
   {
      // map file
      fitsfile.openData(fitsfile.getBitpixOut(), N, naxis_i, !fitsfile.getPrimary());
      // write data
      typename MArray<T, N>::const_iterator i = a.begin();
      fitsfile.writeDataArray(i);
   }
   // close file
   fitsfile.closeData();
   return fitsfile;
}


//@}

/*! \addtogroup ma_fits_io
*/
//@{
//! Class for binary tables IO.
class BinTable
{
private:
   const int nrow_;   
protected:
   BinTable();
   int tfields_;
   char* tformT_;
   size_t* tformr_;
   off_t* tcoloff_;
   vector<string> tforma_;
   double* tscal_;
   double* tzero_;
   off_t theap_;

public:
   BinTable(const int nrow);
   ~BinTable();
   
   int getTfields(const int colno) const;
   char getTformt(const int colno) const;
   size_t getTformr(const int colno) const;   
};

//@}

/*! \addtogroup ma_fits_io
*/
//@{
//! Class to read binary table from FITS extensions.
class FitsBinTableIn : public BinTable, FitsIn
{
public:
   FitsBinTableIn(const FitsIn& other);
   template<class T, int N>
   MArray<T, N> readColumn(const string& ttype, const int startrow = 1, const int endrow = 0);
   template<class T, int N>
   MArray<T, N> readColumn(const int colno, const int startrow = 1, const int endrow = 0);
   template<class T, int N, int BITPIX>
   MArray<T, N> readColumn(const int colno, const int startrow = 1, int endrow = 0);
   template<class T, int N, int BITPIX>
   MArray<T, N> readPColumn(const int colno, const int rowno = 1);
};

//@}

template<class T, int N>
MArray<T, N> FitsBinTableIn::readColumn(const string& ttype, const int startrow, const int endrow)
{
   char buf[9];
   FitsCard* the_card = NULL;
   for(int i = 1; i<=tfields_; ++i){
      sprintf(buf, "TTYPE%-3d", i);
      the_card = findCardInList(string(buf), extension_);
      if(the_card != NULL){
         if(the_card->getString() == ttype)
            return readColumn<T, N>(i, startrow, endrow);
      }
   }
   throw FitsException( string("BINTABLE extension holds no TTYPE = " + ttype + string(".")) );
}

template<class T, int N>
MArray<T, N> FitsBinTableIn::readColumn(const int colno, const int startrow, const int endrow)
{
   if( (colno < 1) || (colno > tfields_) )
      throw FitsException( string("Requested column out of range.") );
   if( tformr_[colno - 1] == 0)
      throw FitsException( string("Requested column is empty (repeat count explicitely set to zero).") );
   
   const char tformT = tformT_[colno - 1];
   MArray<T, N> a;
   switch(tformT){
      case 'L': a.makeReference(readColumn<T, N, 8>(colno, startrow, endrow)); break;
      case 'B': a.makeReference(readColumn<T, N, 8>(colno, startrow, endrow)); break;
      case 'A': a.makeReference(readColumn<T, N, 8>(colno, startrow, endrow)); break;
      case 'I': a.makeReference(readColumn<T, N, 16>(colno, startrow, endrow)); break;
      case 'J': a.makeReference(readColumn<T, N, 32>(colno, startrow, endrow)); break;
      case 'E': a.makeReference(readColumn<T, N, -32>(colno, startrow, endrow)); break;
      case 'D': a.makeReference(readColumn<T, N, -64>(colno, startrow, endrow)); break;
      case 'P': {
         if( tformr_[colno - 1] > 1)
            throw FitsException( string("Binary table TFORM type P with illegal repeat (> 1).") );
         if( (endrow != 0) && (endrow != startrow) )
            throw FitsException( string("Binary table variable length arrays can be read only one at a time into MArrays.") );
         switch(tforma_[colno - 1][0])
         {
            case 'L': a.makeReference(readPColumn<T, N, 8>(colno,   startrow)); break;
            case 'B': a.makeReference(readPColumn<T, N, 8>(colno,   startrow)); break;
            case 'A': a.makeReference(readPColumn<T, N, 8>(colno,   startrow)); break;
            case 'I': a.makeReference(readPColumn<T, N, 16>(colno,  startrow)); break;
            case 'J': a.makeReference(readPColumn<T, N, 32>(colno,  startrow)); break;
            case 'E': a.makeReference(readPColumn<T, N, -32>(colno, startrow)); break;
            case 'D': a.makeReference(readPColumn<T, N, -64>(colno, startrow)); break;
            default:
               throw FitsException(string("Binary table TFORM type P with unsupported variable length type."));
         }
      }
         break;
      case 'X':{
         const size_t bytlen = (tformr_[colno-1] / 8) + 1;
         if(sizeof(T) < bytlen)
            throw FitsException(string("MArray value type to small to hold bitfield Binary Table column."));
         switch(bytlen){
            case 1: a.makeReference(readColumn<T, N, 8>(colno, startrow, endrow)); break;
            case 2: a.makeReference(readColumn<T, N, 16>(colno, startrow, endrow)); break;
            case 3:
               throw FitsException(string("Cannot read binary table TFORM type X wider than 16 bits but smaller than 25 bits. Please complain to cag@usm.lmu.de.")); break;
            case 4: a.makeReference(readColumn<T, N, 32>(colno, startrow, endrow)); break;
            default:
               throw FitsException(string("Cannot read binary table TFORM type X wider than 32 bits. Please complain to cag@usm.lmu.de."));
         }
      }
         break;
      default:
         throw FitsException(string("Binary table TFORM type not yet implemented. Please complain to cag@usm.lmu.de."));
   }
   return a;
}

template<class T, int N, int BITPIX>
MArray<T, N> FitsBinTableIn::readColumn(const int colno, const int startrow, int endrow)
{
   if( (endrow == 0) || (endrow > naxis_array_[1]) )
      endrow = naxis_array_[1];
   if( endrow < startrow )
      throw FitsException( string("Requested end row < start row.") );
   if( (startrow < 1) || (startrow > naxis_array_[1]) )
      throw FitsException( string("Requested start row out of range.") );

   int dim_ [N];
   uint8_t* const sourcearray = begin() + ((startrow-1) * naxis_array_[0]) + tcoloff_[colno - 1];
   uint8_t* sourceptr = sourcearray;
   const size_t tformr = tformr_[colno-1];
   const size_t collen = endrow - startrow + 1;
   if(tformr > 1){
      // check on multi-dim array convention
      char buf[9];
      FitsCard* the_card = NULL;
      sprintf(buf, "TDIM%-4d", colno);
      the_card = findCardInList(string(buf), extension_);
      if(the_card != NULL){ // we have a multidim array
         string tdimstr = the_card->getString();
         if(tdimstr[0] != '(' || tdimstr[tdimstr.size()-1] != ')')
            throw FitsException("TDIM has unknown syntax (not '(x,y,..)'.)");
         // TDIM parser starts here!
         tdimstr.erase(0, 1);
         int i=0, l=1;
         string token;
         string::size_type end = 1;
         // comma separated string of ints
         while( end != tdimstr.npos ){
            end = tdimstr.find_first_of( ",", 0 );
            if( end != tdimstr.npos ){
               token = tdimstr.substr( 0, end );
               tdimstr.erase( 0, end+1 );
            }
            else{
               token = tdimstr.substr( 0, tdimstr.length() );
               tdimstr.erase(0, tdimstr.length() );
            }
            const int dimlen = strtol( token.c_str(), (char **)NULL, 10 );
            if( i == N )
               dim_[i-1] *= dimlen;
            else
               dim_[i++] = dimlen;
            l *= dimlen;
         }
         if((size_t)l != tformr)
            throw(FitsException("TDIM value does not correspond to TFORM r setting in binary table."));
         if(collen > 1){
            if( i == N )
               dim_[i-1] *= collen;
            else
               dim_[i++] = collen;
         }
         while(i<N)dim_[i++] = 1;
      }
      else{ // no multidim array
         dim_[0] = tformr;
         if(N==1) dim_[0] *= collen;
         if(N>1) dim_[1] = collen;
         if(N>2) for(int i = 2; i<N; dim_[i++] = 1);
      }
   }
   else{
      dim_[0] = collen;
      if(N>1) for(int i = 1; i<N; dim_[i++] = 1);
   }
   const size_t destsize = tformr * collen;
   T* const destarray = new T [destsize];
   T* const destend = destarray + destsize;
   T* destptr = destarray;
   const size_t bytlinlength = tcoloff_[colno] - tcoloff_[colno-1]; // = sizeof(typeof(tformT_[colno-1])) * tformr_[colno-1] 

   const double tscal = tscal_[colno-1]; const int itscal = int(tscal);
   const double tzero = tzero_[colno-1]; const int itzero = int(tzero);
   const bool notany = ((tscal == 1.0) && (tzero == 0.0));
   const bool tint   = ( (tscal == double(itscal)) &&
                         (tzero == double(itzero)) );
   while(destptr < destend)
   {
      if(notany)
         destptr = readData<T, BITPIX>(destptr, sourceptr, bytlinlength);
      else{
         if(tint)
            destptr = readData<T, BITPIX>(destptr, sourceptr, bytlinlength,
                                          itscal, itzero);
         else
            destptr = readData<T, BITPIX>(destptr, sourceptr, bytlinlength,
                                          tscal, tzero);
      }
      sourceptr += naxis_array_[0];
   }
   return MArray<T, N>(destarray, dim_);
}

// extract a variable length "column" of the heap
template<class T, int N, int BITPIX>
MArray<T, N> FitsBinTableIn::readPColumn(const int colno, const int rowno)
{
   if( (rowno < 1) || (rowno > naxis_array_[1]) )
      throw FitsException( string("Requested row for variable length array pointer out of range.") );

   uint8_t* const srcptrptr = begin() + ((rowno-1) * naxis_array_[0]) + tcoloff_[colno - 1];
   const uint32_t nelem = sData<uint32_t, 32>::read(srcptrptr);
   const uint32_t offelem = sData<uint32_t, 32>::read(srcptrptr + sizeof(uint32_t));
   
   int dim_ [N];
   uint8_t* const sourceptr = begin() + theap_ + offelem;
   dim_[0] = nelem;
   if(N>1) for(int i = 1; i<N; dim_[i++] = 1);

   T* const destptr = new T [nelem];

   const double tscal = tscal_[colno-1]; const int itscal = int(tscal);
   const double tzero = tzero_[colno-1]; const int itzero = int(tzero);
   const bool notany = ((tscal == 1.0) && (tzero == 0.0));
   const bool tint   = ( (tscal == double(itscal)) &&
                         (tzero == double(itzero)) );
   const size_t bytlinlength = nelem * ( labs( long(BITPIX / 8) ) );
   if(notany)
      readData<T, BITPIX>(destptr, sourceptr, bytlinlength);
   else{
      if(tint)
         readData<T, BITPIX>(destptr, sourceptr, bytlinlength,
                             itscal, itzero);
      else
         readData<T, BITPIX>(destptr, sourceptr, bytlinlength,
                             tscal, tzero);
   }
   return MArray<T, N>(destptr, dim_);
}
   

}

#endif
