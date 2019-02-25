/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits.h 541 2014-07-09 17:01:12Z drory $ 
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

#ifndef __FITS_H__
#define __FITS_H__

#include <ltl/config.h>

// open file
#include <sys/types.h>
#include <sys/stat.h>

// memcpy
#include <cstring>

// ptrdiff_t
#include <cstddef>

// only for debugging
//#include <stdio.h>
//#include <errno.h>

#include <vector> // for FitsExtensionIn object

#include <ltl/misc/exceptions.h>
#include <ltl/io/fits_header.h>
#include <ltl/io/fits_data.h>

#include <ltl/util/region.h>

using std::string;
using std::size_t;
using std::ptrdiff_t;
using std::vector;

//using util::Region; // bad idea to have "using" in a library
using util::UException;

namespace ltl {

//! FitsIO mother class: things FitsIn and FitsOut have in common
class FitsIO
{
   private:
      //! Default Constructor is private to avoid its use.
      FitsIO()
      { }  // no default constructor!

   protected:
      //! Filename associated with FitsIn/Out object.
      string filename_;
      //! Pointer to in-memory fits file image
      unsigned char*	fitsinmemptr_;
      //! Length of in-memory fits file
      size_t			fitsinmemlength_;
      //! In-memory fits file image allocation data
      unsigned char**	ptrfitsinmemptr_;
      size_t*			ptrfitsinmemlength_;
      //! Pointer to data segment map.
      unsigned char* fitsdataptr_;
      //! Pointer to actual byte in data segment for per pixel I/O.
      unsigned char* fitsstreamptr_;
      //! Length of data segment map, multiple of page size.
      size_t fitsmaplength_;

      //! Pointer to Region (if set).
      util::Region* regionptr_;
      //! Offset to determine first pixel of selected region.
      off_t fitsregionoffset_;

      //! Test if dimension dim of region is "retrievable".
      void checkRegion(const size_t dim) const throw(FitsException);

   public:
      //! Set ltl::FitsIO::filename_ and initialize other members.
      FitsIO(unsigned char* inmemptr, size_t inmemlen);
      FitsIO(unsigned char** inmemptr, size_t* inmemlen);
      FitsIO(const string &path);
      //! Return ltl::FitsIO::filename_ .
      string getFilename() const
      { return filename_; }
      //! Return ltl::FitsIO::fitsstreamptr_ .
      unsigned char* streampos() const
      { return fitsstreamptr_; }

      //@{
      //! Get number of pixels in selected region.
      size_t getRegionLength() const;
      //! Get length along dimension dim of selected region in pixels.
      size_t getRegionLength(const size_t dim) const;
      //! Get start pixel coordinate in dimension dim of selected region.
      int getRegionStart(const size_t dim) const;
      //! Get end pixel coordinate in dimension dim of selected region.
      int getRegionEnd(const size_t dim) const;
      //! Return copy of selected region.
      util::Region getRegion() const;
      //@}
};

/*! \addtogroup ma_fits_io
*/
//@{

//! Reading FITS data segment, reading and modifying FITS header.
class FitsIn : public FitsHeader, public FitsIO
{
   private:
      //! Map selected region of data segment.
      void openData();
      //! Test if dimension dim of region is "retrievable".
      void checkRegion(const size_t dim) const throw(FitsException);
      friend class FitsOut;

   protected:
      //! Boolean indicating if header is not parsed.
      bool ignore_hd_;
      //! Boolean indicating BSCALE and BZERO being integer.
      bool bint_;

      //! construct a extension object
      FitsIn(const string& path,
             const bool quiet_please,
             const off_t startoffset);
   public:
      //! Construct a FitsIn object from in memory file image
	  FitsIn(unsigned char* inmemptr, size_t inmemlen,
             const bool quiet_please = false,
             const bool ignore_header = false);

      //! Construct a FitsIn object from C string filename.
      FitsIn(const char * path,
             const bool quiet_please = false,
             const bool ignore_header = false);

      //! Construct a FitsIn object from C++ string filename.
      FitsIn(const string & path,
             const bool quiet_please = false,
             const bool ignore_header = false);

      FitsIn(const char * path, const util::Region & freg,
             const bool quiet_please = false,
             const bool ignore_header = false);

      FitsIn(const string & path, const util::Region & freg,
             const bool quiet_please = false,
             const bool ignore_header = false);

      //! Construct as copy from \e other
      FitsIn(const FitsIn& other);

      virtual ~FitsIn();

      //! Unmap data segment, free address space.
      void freeData();

      //! Write FITS file geometry and class interna to out stream.
      virtual void describeSelf( std::ostream& os );

      //@{
      unsigned char* begin();
      unsigned char* end();
      void resetPosition();
      void setPosition(off_t offset);
      ptrdiff_t getPosition();
      //@}

      //@{
      //! Select region of interest in data segment.
      void setRegion(const util::Region& fregion) throw(FitsException);
      //! Reset region of interest in data segment to full region.
      void resetRegion();
      //@}

      //! Return copy of FITS header.
      FitsHeader getHeader() const;

      //! Return input object for next FITS extension when present.
      virtual FitsIn getNextExtension();

      //@{
      //! Return array holding preselected region of data segment
      template<class T>
      T* getRegionArray(const util::Region& reg);

      //! Fill container via iterator i with region reg of data segment
      template<class T>
      void readRegionArray(T& i, const util::Region& reg);

      //! Fill container via iterator i with preselected region of data segment
      template<class T>
      void readRegionArray(T& i);

      //! Fill container via iterator i with data segment
      template<class T>
      void readDataArray(T& i);

      //! Return array holding data segment
      template<class T>
      T* getDataArray(const T);

      //! Read next pixel of datasegment with per pixel read
      template<class T>
      T getNextPixel(const T);
      //@}
};

//@}


template<class T>
T* FitsIn::getRegionArray(const util::Region& reg)
{
   setRegion(reg);
   T* const retptr = getDataArray<T>();
   resetRegion();
   return retptr;
}

template<class T>
void FitsIn::readRegionArray(T& i, const util::Region& reg)
{
   setRegion(reg);
   readRegionArray(i);
   resetRegion();
}


template<class T>
void FitsIn::readRegionArray(T& i)
{
   if(regionptr_ == NULL)
      throw FitsException("no region set");
   
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
         ltl::readRegionArray<T, LTL_FITS_BITPIX_CHAR>(i, begin(),
                                                       getFullRegion(), *regionptr_,
                                                       getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_SHORT:
         ltl::readRegionArray<T, LTL_FITS_BITPIX_SHORT>(i, begin(),
                                                        getFullRegion(), *regionptr_,
                                                        getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_INT:
         ltl::readRegionArray<T, LTL_FITS_BITPIX_INT>(i, begin(),
                                                      getFullRegion(), *regionptr_,
                                                      getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_FLOAT:
         ltl::readRegionArray<T, LTL_FITS_BITPIX_FLOAT>(i, begin(),
                                                        getFullRegion(), *regionptr_,
                                                        getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_DOUBLE:
         ltl::readRegionArray<T, LTL_FITS_BITPIX_DOUBLE>(i, begin(),
                                                         getFullRegion(), *regionptr_,
                                                         getBscale(), getBzero());
         break;
      default: throw FitsException("illegal BITPIX");
   }
}


template<class T>
T* FitsIn::getDataArray(const T)
{
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
         if(regionptr_ == NULL)
            return ltl::getDataArray<T, LTL_FITS_BITPIX_CHAR>(begin(), getDataLength(),
                                                              getBscale(), getBzero());
         else
            return ltl::getRegionArray<T, LTL_FITS_BITPIX_CHAR>(begin(), getFullRegion(), *regionptr_,
                                                                getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_SHORT:
         if(regionptr_ == NULL)
            return ltl::getDataArray<T, LTL_FITS_BITPIX_SHORT>(begin(), getDataLength(),
                                                               getBscale(), getBzero());
         else
            return ltl::getRegionArray<T, LTL_FITS_BITPIX_SHORT>(begin(), getFullRegion(), *regionptr_,
                                                                 getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_INT:
         if(regionptr_ == NULL)
            return ltl::getDataArray<T, LTL_FITS_BITPIX_INT>(begin(), getDataLength(),
                                                             getBscale(), getBzero());
         else
            return ltl::getRegionArray<T, LTL_FITS_BITPIX_INT>(begin(), getFullRegion(), *regionptr_,
                                                               getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_FLOAT:
         if(regionptr_ == NULL)
            return ltl::getDataArray<T, LTL_FITS_BITPIX_FLOAT>(begin(), getDataLength(),
                                                               getBscale(), getBzero());
         else
            return ltl::getRegionArray<T, LTL_FITS_BITPIX_FLOAT>(begin(), getFullRegion(), *regionptr_,
                                                                 getBscale(), getBzero());
         break;
      case LTL_FITS_BITPIX_DOUBLE:
         if(regionptr_ == NULL)
            return ltl::getDataArray<T, LTL_FITS_BITPIX_DOUBLE>(begin(), getDataLength(),
                                                                getBscale(), getBzero());
         else
            return ltl::getRegionArray<T, LTL_FITS_BITPIX_DOUBLE>(begin(), getFullRegion(), *regionptr_,
                                                                  getBscale(), getBzero());
         break;
      default: throw FitsException("illegal BITPIX");
   }
   return NULL;
}

template<class T>
void FitsIn::readDataArray(T& i)
{
   if(regionptr_ != NULL)
      readRegionArray(i);
   else
   {
      switch(bitpix_)
      {
         case LTL_FITS_BITPIX_CHAR:
            ltl::readDataArray<T, LTL_FITS_BITPIX_CHAR>(i, begin(), getDataLength(),
                                                        getBscale(), getBzero());
            break;
         case LTL_FITS_BITPIX_SHORT:
            ltl::readDataArray<T, LTL_FITS_BITPIX_SHORT>(i, begin(), getDataLength(),
                                                         getBscale(), getBzero());
            break;
         case LTL_FITS_BITPIX_INT:
            ltl::readDataArray<T, LTL_FITS_BITPIX_INT>(i, begin(), getDataLength(),
                                                       getBscale(), getBzero());
            break;
         case LTL_FITS_BITPIX_FLOAT:
            ltl::readDataArray<T, LTL_FITS_BITPIX_FLOAT>(i, begin(), getDataLength(),
                                                         getBscale(), getBzero());
            break;
         case LTL_FITS_BITPIX_DOUBLE:
            ltl::readDataArray<T, LTL_FITS_BITPIX_DOUBLE>(i, begin(), getDataLength(),
                                                          getBscale(), getBzero());
            break;
         default: throw FitsException("illegal BITPIX");
      }
   }
}



// Read next pixel of datasegment with per pixel read
template<class T>
T FitsIn::getNextPixel(const T)
{
   if(streampos() == NULL)
      resetPosition();   
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
      {
         const T value =
            sData<T, LTL_FITS_BITPIX_CHAR>::read(fitsstreamptr_,
                                                 bscale_, bzero_);
         ppDataPointer<LTL_FITS_BITPIX_CHAR>(fitsstreamptr_);
         return value;
      }
      break;
      case LTL_FITS_BITPIX_SHORT:
      {
         const T value =
            sData<T, LTL_FITS_BITPIX_SHORT>::read(fitsstreamptr_,
                                                  bscale_, bzero_);
         ppDataPointer<LTL_FITS_BITPIX_SHORT>(fitsstreamptr_);
         return value;
      }
      break;
      case LTL_FITS_BITPIX_INT:
      {
         const T value =
            sData<T, LTL_FITS_BITPIX_INT>::read(fitsstreamptr_,
                                                bscale_, bzero_);
         ppDataPointer<LTL_FITS_BITPIX_INT>(fitsstreamptr_);
         return value;
      }
      break;
      case LTL_FITS_BITPIX_FLOAT:
      {
         const T value =
            sData<T, LTL_FITS_BITPIX_FLOAT>::read(fitsstreamptr_,
                                                  bscale_, bzero_);
         ppDataPointer<LTL_FITS_BITPIX_FLOAT>(fitsstreamptr_);
         return value;
      }
      break;
      case LTL_FITS_BITPIX_DOUBLE:
      {
         const T value =
            sData<T, LTL_FITS_BITPIX_DOUBLE>::read(fitsstreamptr_,
                                                   bscale_, bzero_);
         ppDataPointer<LTL_FITS_BITPIX_DOUBLE>(fitsstreamptr_);
         return value;
      }
      break;
   }
   throw FitsException("illegal BITPIX");
   return T(0);
}

/*! \addtogroup ma_fits_io
*/
//@{

//! Writing FITS files
class FitsOut : public FitsHeader, public FitsIO
{
   protected:
      //! Boolean indicating if "junk" FITS cards are also written to header.
      bool ignorejunk_;
      //! BITPIX setting for FITS file.
      int bitpixout_;
      //! Shortcut to ORIGIN key.
      string origin_;

      //! Convert Region to Naxis array discarding dims with length == 1.
      int* region2Naxis(const util::Region& region, int& newnaxis) const;
      //! Create "empty" FITS file, i.e. header + zero data segment. Returns file descriptor.
      int setGeometry(const int newbitpix,
                      const int newnaxis,
                      const int* newnaxis_i,
                      const bool append = false);
      void checkinmemalloc(const off_t write_offset, const int bytestobewritten);

   public:

      FitsOut();
      //! Construct a FitsOut object to an in memory file image
      FitsOut(unsigned char** inmemptr, size_t* inmemlen);
      //! Construct FITS file object with filename \e path.
      FitsOut(const string& path);
      FitsOut(const char* path);
      //! Construct FITS file object copying FITS keys from header.
      FitsOut(const string& path, const FitsHeader& header,
              const bool quiet_please = false, const bool ign_junk = false);
      //! Construct FITS file object copying FITS keys from header.
      FitsOut(const char* path, const FitsHeader& header,
              const bool quiet_please = false, const bool ign_junk = false);

      virtual ~FitsOut();

      //! Write FITS file geometry and class interna to out stream.
      virtual void describeSelf( std::ostream& os );

   //! Write FITS file header and create suitable data segment map.
   void openData(const int newbitpix,
                 const int newnaxis,
                 const int * newnaxis_i, const bool append = false) throw(FitsException);
   //! Write FITS file header and create region sized data segment map.
   void openData(const int newbitpix,
                 const util::Region& region, const bool append = false) throw(FitsException);
   //! Create "empty" FITS file, i.e. header + zero data segment.
   void setGeometry(const int newbitpix, const util::Region& region,
                    const bool append = false);
      //! Set Region for selective access to parts of data segment .
      void setRegion(const util::Region& region) throw(FitsException);
      //! Reset Region, if set.
      void resetRegion()
      { closeData(); }
      //! Return true if region is set.
      bool isRegion() const
      { return (regionptr_ != NULL); }

      //! Finish FITS file, unmap data segment.
      void closeData();
      //! Copy data (and BSCALE and BZERO) from infile.
      void copyData(FitsIn& infile, const bool append = false);

      unsigned char* begin() const;
      unsigned char* end() const;
      void resetPosition();
      void setPosition(off_t offset);
      ptrdiff_t getPosition() const;

      //! Set output filename according to argument.
      void setFilename(const string& path);
      //! Override automatic BITPIX setting with argument.
      void setBitpixOut(const int bpo) throw(FitsException);
      //! Set FITS key ORIGIN according to argument.
      void setOrigin(const string& orig);
      //! Public method to read ltl::FitsOut::bitpixout_
      int getBitpixOut() const;
      //! Public method to read ltl::FitsOut::origin_
      string getOrigin() const;

      //! Fill data segment with container via iterator i.
      template<class T>
      void writeDataArray(T& i);
      //! Fill region of data segment with container via iterator i.
      template<class T>
      void writeRegionArray(T& i);

      //! Per pixel writing, set next pixel to value.
      template<class T>
      void setNextPixel(const T value);

   protected:
      //! Erase BSCALE, BZERO, BLOCKED, EPOCH and ORIGIN.
      void eraseObsolete();
      //! Set mandatory FITS keywords according to arguments.
      void resetMandatories(const int newbitpix, const int newnaxis,
                            const int* newnaxis_i) throw(FitsException);
      //! Erase potentially missleading FITS array keywords.
      void resetArrayKeys();
};

//@}

template<class T>
void FitsOut::writeDataArray(T& i)
{
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
         writeData<T, LTL_FITS_BITPIX_CHAR>(begin(), getDataLength(), i);
         break;
      case LTL_FITS_BITPIX_SHORT:
         writeData<T, LTL_FITS_BITPIX_SHORT>(begin(), getDataLength(), i);
         break;
      case LTL_FITS_BITPIX_INT:
         writeData<T, LTL_FITS_BITPIX_INT>(begin(), getDataLength(), i);
         break;
      case LTL_FITS_BITPIX_FLOAT:
         writeData<T, LTL_FITS_BITPIX_FLOAT>(begin(), getDataLength(), i);
         break;
      case LTL_FITS_BITPIX_DOUBLE:
         writeData<T, LTL_FITS_BITPIX_DOUBLE>(begin(), getDataLength(), i);
         break;
      default: throw FitsException("illegal BITPIX");
   }
}

template<class T>
void FitsOut::writeRegionArray(T& i)
{
   if(regionptr_ == NULL)
      throw FitsException("no region set");
   
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
         ltl::writeRegionArray<T, LTL_FITS_BITPIX_CHAR>(begin(), i, 
                                                       getFullRegion(), *regionptr_);
         break;
      case LTL_FITS_BITPIX_SHORT:
         ltl::writeRegionArray<T, LTL_FITS_BITPIX_SHORT>(begin(), i, 
                                                        getFullRegion(), *regionptr_);
         break;
      case LTL_FITS_BITPIX_INT:
         ltl::writeRegionArray<T, LTL_FITS_BITPIX_INT>(begin(), i, 
                                                      getFullRegion(), *regionptr_);
         break;
      case LTL_FITS_BITPIX_FLOAT:
         ltl::writeRegionArray<T, LTL_FITS_BITPIX_FLOAT>(begin(), i, 
                                                        getFullRegion(), *regionptr_);
         break;
      case LTL_FITS_BITPIX_DOUBLE:
         ltl::writeRegionArray<T, LTL_FITS_BITPIX_DOUBLE>(begin(), i,
                                                         getFullRegion(), *regionptr_);
         break;
      default: throw FitsException("illegal BITPIX");
   }
}

template<class T>
void FitsOut::setNextPixel(const T value)
{
   if(streampos() == NULL)
      resetPosition();   
   switch(bitpix_)
   {
      case LTL_FITS_BITPIX_CHAR:
      {
         sData<T, LTL_FITS_BITPIX_CHAR>::write(fitsstreamptr_, value);
         ppDataPointer<LTL_FITS_BITPIX_CHAR>(fitsstreamptr_);
      }
      break;
      case LTL_FITS_BITPIX_SHORT:
      {
         sData<T, LTL_FITS_BITPIX_SHORT>::write(fitsstreamptr_, value);
         ppDataPointer<LTL_FITS_BITPIX_SHORT>(fitsstreamptr_);
      }
      break;
      case LTL_FITS_BITPIX_INT:
      {
         sData<T, LTL_FITS_BITPIX_INT>::write(fitsstreamptr_, value);
         ppDataPointer<LTL_FITS_BITPIX_INT>(fitsstreamptr_);
      }
      break;
      case LTL_FITS_BITPIX_FLOAT:
      {
         sData<T, LTL_FITS_BITPIX_FLOAT>::write(fitsstreamptr_, value);
         ppDataPointer<LTL_FITS_BITPIX_FLOAT>(fitsstreamptr_);
      }
      break;
      case LTL_FITS_BITPIX_DOUBLE:
      {
         sData<T, LTL_FITS_BITPIX_DOUBLE>::write(fitsstreamptr_, value);
         ppDataPointer<LTL_FITS_BITPIX_DOUBLE>(fitsstreamptr_);
      }
      break;
      default: throw FitsException("illegal BITPIX");
   }
}

/*! \addtogroup ma_fits_io
*/
//@{

//! Common methods for FitsExtensionIn/Out.
class FitsExtension{
public:
   FitsExtension() : primary_(true), extno_(0) { }
   //! Return number of actual extension.
   size_t getExtNo() const;
   //! Return status of primary flag, i.e. reading or writing primary header / data.
   bool getPrimary() const;
   //! Reset Extension counter and raise primary flag.
   void resetExtNo();
   //! Increment Extension counter and lower primary flag.
   void incExtNo();
   //! Lower primary flag.
   void unsetPrimary();

protected:
   bool primary_;
   size_t extno_;
};

//! Primary HDU and extensions in one object. Look into test/testfitsextensionio.cpp for examples.
class FitsExtensionIn : public FitsIn, public FitsExtension
{
   public:
      //! Construct a FitsExtensionIn object from C string filename.
      FitsExtensionIn(const char * path,
                      const bool quiet_please = false,
                      const bool ignore_header = false);

      //! Construct a FitsExtensionIn object from C++ string filename.
      FitsExtensionIn(const string & path,
                      const bool quiet_please = false,
                      const bool ignore_header = false);

      virtual ~FitsExtensionIn();

     /*! Return FitsIn object holding extension of type \e XTENSION with name \e EXTNAME
         and optional version \e EXTVER and level \e EXTLEVEL (get by name).
     */
      FitsIn getExtension(const string& xtension, const string& extname,
                          const int extver = 0, const int extlevel = 0);
      //! Return FitsIn object holding \e extno 's extension (get by sequential number).
      FitsIn getExtension(const size_t extno);
      //! Return the next extension.
      virtual FitsIn getNextExtension();

   private:
      vector<FitsIn> extension_;
};

// utilise FitsHeader::data_offset_
// put Extension write methods already hidden into FitsOut
// just "disclose" them in FitsExtensionOut
// openData(..., const bool append = false);
//! Create a FITS extension object. Look into test/testfitsextensionio.cpp for examples.
class FitsExtensionOut : public FitsOut, public FitsExtension
{
public:
   FitsExtensionOut(const string& path);
   FitsExtensionOut(const char* path);
   FitsExtensionOut(const string& path, const FitsHeader& header,
                    const bool quiet_please = false, const bool ign_junk = false);
   FitsExtensionOut(const char* path, const FitsHeader& header,
                    const bool quiet_please = false, const bool ign_junk = false);
   virtual ~FitsExtensionOut();

   //! dump header to file, i.e. either primary or of next extension
   void flushHeader();
};
//@}

/*! \relates ltl::FitsExtensionOut
 * Helper class to write FitsHeaders without data segment to FITS containing extension.
 */
class emptyData_
{ 
public:
   emptyData_() { }
   ~emptyData_() { }
};

extern const emptyData_ emptyData;

/*! \addtogroup ma_fits_io
*/
//@{

/*! \relates ltl::FitsIn
  Overload of global \>\>, raw read of complete data segment.
*/
template<class T>
inline FitsIn& operator>>(FitsIn& fitsfile, T& a)
{
   a = fitsfile.getNextPixel(T(0));
   return fitsfile;
}

/*! \relates ltl::FitsIO
  Copy (selected region of) data segment from \e infile to \e outfile.
*/
template<>
inline FitsIn& operator>> <FitsOut>(FitsIn& infile,
                                    FitsOut& outfile)
{
   outfile.copyData(infile);
   return infile;
}

/*! \relates ltl::FitsOut
  Overload of global \<\<.
*/
template<class T>
inline FitsOut& operator<<(FitsOut& fitsfile, const T& a)
{
   fitsfile.setNextPixel(a);
   return fitsfile;
}

/*! \relates ltl::FitsIO
  Copy (selected region of) data segment from \e infile to \e outfile.
*/
//template<>
//inline FitsOut& operator<< <FitsIn>(FitsOut& outfile,
inline FitsOut& operator<< (FitsOut& outfile,
                            FitsIn& infile)
{
   outfile.copyData(infile);
   return outfile;
}

/*! \relates ltl::FitsIO
  Copy data of extension from \e infile to \e outfile.
  Cycles through extensions.
*/
// template<>
// inline FitsOut& operator<< <FitsExtensionIn>(FitsOut& outfile,
inline FitsOut& operator<< (FitsOut& outfile,
                            FitsExtensionIn& infile)
{
   if( infile.getPrimary() )
   {
      outfile.copyData(infile);      
      infile.unsetPrimary();
   }
   else
   {
      try
      {
         FitsIn inext( infile.getNextExtension() );
         outfile.copyData(inext);
      }
      catch(FitsException e)
      {
         infile.resetExtNo();
         outfile << infile;
      }
   }
   return outfile;
}

/*! \relates ltl::FitsIO
  Write header without data segment (NAXIS=0) to \e outfile.
*/
inline FitsExtensionOut& operator<< (FitsExtensionOut& outfile,
                                     const emptyData_&)
{
   outfile.openData( 8, 0, NULL, !outfile.getPrimary() );
   outfile.incExtNo();
   return outfile;
}

/*! \relates ltl::FitsIO
  Copy (selected region of) data segment from \e infile to \e outfile.
*/
inline FitsExtensionOut& operator<< (FitsExtensionOut& outfile,
                            FitsIn& infile)
{
   outfile.copyData(infile, !outfile.getPrimary());
   outfile.incExtNo();
   return outfile;
}
//@}

}

#endif
