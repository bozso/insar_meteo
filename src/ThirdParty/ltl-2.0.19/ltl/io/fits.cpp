/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits.cpp 544 2014-08-01 14:48:03Z rbryant $
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

#include <ltl/fits.h>
#include <iostream>

// FitsIn implementation

namespace ltl {

void FitsIO::checkRegion(const size_t dim) const throw(FitsException)
{
   if(regionptr_ == NULL)
      throw FitsException("request for unset region");
   if( dim < 1 )
      throw FitsException("request for invalid region parameter");
}

// Constructor of FitsIO
FitsIO::FitsIO(unsigned char* inmemptr, size_t inmemlen) :
   filename_(string()),
   fitsinmemptr_(inmemptr), fitsinmemlength_(inmemlen),
   fitsdataptr_(NULL), fitsstreamptr_(NULL), fitsmaplength_((size_t)0),
   regionptr_(NULL), fitsregionoffset_(0)
{ }

FitsIO::FitsIO(unsigned char** inmemptr, size_t* inmemlen) :
   filename_(string()),
   fitsinmemptr_(*inmemptr), fitsinmemlength_(*inmemlen),
   ptrfitsinmemptr_(inmemptr), ptrfitsinmemlength_(inmemlen),
   fitsdataptr_(NULL), fitsstreamptr_(NULL), fitsmaplength_((size_t)0),
   regionptr_(NULL), fitsregionoffset_(0)
{ }

FitsIO::FitsIO(const string& path) :
   filename_(path),
   fitsdataptr_(NULL), fitsstreamptr_(NULL), fitsmaplength_((size_t)0),
   regionptr_(NULL), fitsregionoffset_(0)
{ }

size_t FitsIO::getRegionLength() const
{
   if(regionptr_) return regionptr_->getLength();
   return 0;
}

size_t FitsIO::getRegionLength(const size_t dim) const
{
   checkRegion(dim);
   return regionptr_->getLength(dim);
}

int FitsIO::getRegionStart(const size_t dim) const
{
   checkRegion(dim);
   return regionptr_->getStart(dim);
}

int FitsIO::getRegionEnd(const size_t dim) const
{
   checkRegion(dim);
   return regionptr_->getEnd(dim);
}

util::Region FitsIO::getRegion() const
{
   return *regionptr_;
}

void FitsIn::checkRegion(const size_t dim) const throw(FitsException)
{
   FitsIO::checkRegion(dim);
   if( dim > size_t(naxis_) )
      throw FitsException("request for invalid region parameter");
}

// FitsIn: constructors / destructor
FitsIn::FitsIn(unsigned char* inmemptr, size_t inmemlen,
               const bool quiet_please,
               const bool ignore_header) :
   FitsHeader(inmemptr, inmemlen, quiet_please, ignore_header), FitsIO(inmemptr, inmemlen),
   ignore_hd_(ignore_header),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{ }

FitsIn::FitsIn(const char* path,
               const bool quiet_please,
               const bool ignore_header) :
   FitsHeader(path, quiet_please, ignore_header), FitsIO(string(path)),
   ignore_hd_(ignore_header),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{ }

FitsIn::FitsIn(const string& path,
               const bool quiet_please,
               const bool ignore_header) :
   FitsHeader(path, quiet_please, ignore_header), FitsIO(path),
   ignore_hd_(ignore_header),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{ }

FitsIn::FitsIn(const char* path, const util::Region& freg,
               const bool quiet_please,
               const bool ignore_header) :
   FitsHeader(path, quiet_please, ignore_header), FitsIO(string(path)),
   ignore_hd_(ignore_header),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{
   setRegion(freg);
}

FitsIn::FitsIn(const string& path, const util::Region& freg,
               const bool quiet_please,
               const bool ignore_header) :
   FitsHeader(path, quiet_please, ignore_header), FitsIO(path),
   ignore_hd_(ignore_header),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{
   setRegion(freg);
}

FitsIn::FitsIn(const FitsIn& other) :
   FitsHeader(other, other.shutup_, other.ignore_hd_),
   FitsIO(other.filename_),
   ignore_hd_(other.ignore_hd_),
   bint_(other.bint_)
{
   if(other.regionptr_ != NULL)
      regionptr_ = new util::Region(*(other.regionptr_));
}

FitsIn::FitsIn(const string& path,
               const bool quiet_please,
               const off_t startoffset) :
   FitsHeader(path, quiet_please, false, startoffset), FitsIO(path),
   ignore_hd_(false),
   bint_((bscale_ == double(int(bscale_))) &&
         (bzero_  == double(int(bzero_ ))))
{ }

FitsIn::~FitsIn()
{
   freeData();
   resetRegion();
}

void FitsIn::openData()
{
   if(naxis_ == 0)
      throw FitsException("No data segment associated with header (NAXIS = 0).");

   const FitsCard* hdr_ident = *(mandatory_.begin());
   if( (*hdr_ident).what_id() == XTENSION )
   {
      if( (*hdr_ident).getString() != string("IMAGE") && 
          (*hdr_ident).getString() != string("BINTABLE") )
         throw FitsException( (*hdr_ident).getString() +
                              string(" extension not supported until now."));
   }
   
   const off_t bpix = off_t(bytpix_);
   const off_t doff = data_offset_;
   const off_t psize = off_t(getpagesize());

   off_t regstartoff = 0;
   off_t reglen      = 0;
   
   if(regionptr_ != NULL)
   {
      off_t regendoff   = 0;
      for(int naxiscounter = naxis_; naxiscounter > 1;)
      {
         regstartoff += off_t(regionptr_->getStart(naxiscounter)) - 1;
         regendoff += off_t(regionptr_->getEnd(naxiscounter)) - 1;
         --naxiscounter;
         const off_t nx_i = off_t(getNaxis(naxiscounter));
         regstartoff *= nx_i;
         regendoff *= nx_i;
      }
      regstartoff += off_t(regionptr_->getStart(1)) - 1;
      regendoff += off_t(regionptr_->getEnd(1)) - 1;
      regstartoff *= bpix;
      regendoff *= bpix;
      reglen = regendoff + bpix - regstartoff;
   }
   else
   {
      reglen = data_length_;
   }
   
   const size_t bytelength = ((doff + regstartoff) % psize) + reglen;
   const off_t fileoff = ((doff + regstartoff) / psize) * psize;

   int	fd;
   bool	inmem = filename_.empty();

   if (!inmem)
   {
      fd = open(filename_.c_str(), O_RDONLY);

	  if (fd < 0)
		 throw FitsException("cannot open file '" +
							  filename_ + "' for reading");
	  fitsdataptr_ = (unsigned char *) mmap( NULL, bytelength,
	                                         PROT_READ, MAP_PRIVATE,
	                                         fd, fileoff );
	  close (fd);
	  if (fitsdataptr_ == (unsigned char *) MAP_FAILED)
	     throw FitsException("cannot map file '" +
	                          filename_ + "' for reading");
   }
   else
   {
	  if (fileoff >= (off_t)fitsinmemlength_)
	     throw FitsException("cannot map in-memory file for reading (fileoff exceeds length of in-memory buffer)");
	  fitsdataptr_ = fitsinmemptr_ + fileoff;
   }

   fitsmaplength_ = bytelength;
   fitsregionoffset_ = fileoff;
}

void FitsIn::freeData()
{
   if(!filename_.empty() && fitsdataptr_)
      munmap((char *)fitsdataptr_, fitsmaplength_);
   fitsdataptr_ = NULL;
   fitsmaplength_ = 0;
   fitsregionoffset_ = 0;
   fitsstreamptr_ = NULL;
}

unsigned char* FitsIn::begin()
{
   if(fitsdataptr_ == NULL)
      openData();
   return fitsdataptr_ + (data_offset_ - fitsregionoffset_);
}

unsigned char* FitsIn::end()
{ 
   return ( begin() + data_length_ );
}

void FitsIn::describeSelf( std::ostream& os )
{
   os << "this-> " << (void *)this << std::endl;
   os << "Fits: file='" << filename_ << "'\n";
   FitsHeader::describeSelf(os);
   os << "Data segment offset: " << data_offset_ << std::endl;
   os << "Data segment length: " << data_length_ << std::endl;
   if(regionptr_ != NULL)
   {
      os << "Region: " << (*regionptr_).toString() << std::endl;
   }
   if(fitsdataptr_ != NULL)
   {
      os << "Memory map @: " << (void *)(fitsdataptr_) << std::endl
         << "File offset: " << fitsregionoffset_ << std::endl;
   }
}

void FitsIn::resetPosition()
{
   fitsstreamptr_ = begin();
}
void FitsIn::setPosition(off_t offset)
{
   fitsstreamptr_ = begin() + offset;
}
ptrdiff_t FitsIn::getPosition()
{
   if(fitsstreamptr_ == NULL)
      fitsstreamptr_ = begin();
   return (fitsstreamptr_ - begin());
}


// set a region for extraction using Region class
void FitsIn::setRegion(const util::Region& fregion) throw(FitsException)
{
   resetRegion();
   testRegion(fregion);
   regionptr_ = new util::Region(fregion);
}

void FitsIn::resetRegion()
{
   if(regionptr_)
   {
      delete regionptr_;
      regionptr_ = NULL;
      freeData();
   }
}

FitsHeader FitsIn::getHeader() const
{
   FitsHeader header(*this);
   return header;
}

// return header of next extension if present
FitsIn FitsIn::getNextExtension()
{
   if( ! extended_ )
      throw FitsException("No extension present");
   const off_t record_length = off_t(PH_C.RECORD_LENGTH);
   const off_t startoffset = data_offset_ +
      ( (data_length_ + record_length - off_t(1)) /
        record_length ) * record_length;
   return FitsIn(filename_, shutup_, startoffset);
}


// FitsOut implementation

// constructors / destructor
FitsOut::FitsOut(unsigned char** inmemptr, size_t* inmemlen) :
   FitsHeader(), FitsIO(inmemptr, inmemlen),
   ignorejunk_(false), bitpixout_(0), origin_("")
{ }

FitsOut::FitsOut() :
   FitsHeader (), FitsIO(string("out.fits")),
   ignorejunk_(false), bitpixout_(0), origin_(string(""))
{ }

FitsOut::FitsOut(const string& path) :
   FitsHeader (), FitsIO(path),
   ignorejunk_(false), bitpixout_(0), origin_("")
{ }

FitsOut::FitsOut(const char* path) :
   FitsHeader (), FitsIO(string(path)),
   ignorejunk_(false), bitpixout_(0), origin_("")
{ }

FitsOut::FitsOut(const string& path, const FitsHeader& header,
                 const bool quiet_please, const bool ign_junk ) :
   FitsHeader(header, quiet_please, false), FitsIO(path),
   ignorejunk_(ign_junk), bitpixout_(0), origin_("")
{
   eraseObsolete();
}

FitsOut::FitsOut(const char* path, const FitsHeader& header,
                 const bool quiet_please, const bool ign_junk ) :
   FitsHeader(header, quiet_please, false), FitsIO(string(path)),
   ignorejunk_(ign_junk), bitpixout_(0), origin_("")
{
   eraseObsolete();
}

FitsOut::~FitsOut()
{
   closeData();
}

void FitsOut::describeSelf( std::ostream& os )
{
   os << "this-> " << (void *)this << std::endl;
   os << "Fits: file='" << filename_ << "'\n";
   FitsHeader::describeSelf(os);
   os << "Data segment offset: " << data_offset_ << std::endl;
   os << "Data segment length: " << data_length_ << std::endl;
   if(regionptr_ != NULL)
   {
      os << "Region: " << (*regionptr_).toString() << std::endl;
   }
   if(fitsdataptr_ != NULL)
   {
      os << "Memory map @: " << (void *)(fitsdataptr_) << std::endl
         << "File offset: " << fitsregionoffset_ << std::endl;
   }
}


// protected members
void FitsOut::eraseObsolete()
{
   eraseCard("BSCALE  ");
   bscale_ = 1.0;
   eraseCard("BZERO   ");
   bzero_ = 0.0;
   eraseCard("BLOCKED ");
   eraseCard("EPOCH   ");
   eraseCard("ORIGIN  ");
}

void FitsOut::resetArrayKeys()
{
   // if float erase BLANK card
   if(bitpix_ < 0)
      eraseCard("BLANK   ");

   // erase CTYPEn, CRPIXn, CRVALn, CDELTn and CROTAn beyond NAXIS n
   //list<FitsCard *>::iterator iter = array_.begin();
   //const list<FitsCard *>::iterator array_end = array_.end();
   //while(iter != array_end)
   //{
   //   if( atoi( (((**iter).getKeyword()).substr(5, 3)).c_str() ) > naxis_)
   //   {
   //      delete *iter;
   //      iter = array_.erase(iter);
   //   }
   //   else
   //      ++iter;
   //}
}

void FitsOut::resetMandatories(const int newbitpix,
                               const int newnaxis,
                               const int* newnaxis_i) throw(FitsException)
{
   // check new bitpix and naxis parameters
   if( (newbitpix != KNOWN.BITPIX_CHAR) &&
       (newbitpix != KNOWN.BITPIX_SHORT) && (newbitpix != KNOWN.BITPIX_INT) &&
       (newbitpix != KNOWN.BITPIX_FLOAT) && (newbitpix != KNOWN.BITPIX_DOUBLE))
      throw FitsException("invalid BITPIX");
   if(newnaxis > KNOWN.NAXIS_HI_LIMIT)
      throw FitsException("NAXIS restricted to < 1000");
   if(newnaxis < KNOWN.NAXIS_LO_LIMIT)
      throw FitsException("NAXIS may not be negative");
   // clear mandatory lists
   clearCardList(mandatory_);
   clearCardList(naxis_i_);
   clearCardList(extension_);
   // reset mandatorys
   if(data_offset_)
      mandatory_.push_back(new FitsStringCard("XTENSION", "IMAGE   ", true));
   else
      mandatory_.push_back(new FitsBoolCard("SIMPLE  ", true, true));
   mandatory_.push_back(new FitsIntCard("BITPIX  ", newbitpix, true));
   bitpix_ = newbitpix;
   bytpix_ = abs(newbitpix) / 8;
   mandatory_.push_back(new FitsIntCard("NAXIS   ", newnaxis, true));
   naxis_ = newnaxis;
   if(newnaxis > 0)
   {
      const string naxislead = string("NAXIS");
      for(int j = 0; j < newnaxis; ++j)
      {
         const int i = j + 1;
         string naxiscounter = naxislead;
         if(i > 99)
         {
            naxiscounter += char( (i/100) + 0x30 );
            naxiscounter += char( ((i%100) / 10) + 0x30 );
            naxiscounter += char( (i%10) + 0x30 );
         }
         else
         {
            if(i > 9)
            {
               naxiscounter += char( (i / 10) + 0x30 );
               naxiscounter += char( (i % 10) + 0x30 );
            }
            else
               naxiscounter += char( i + 0x30 );
            naxiscounter.resize(PH_C.KEYWORD_LENGTH, ' ');
         }
         naxis_i_.push_back(new FitsIntCard(naxiscounter,
                                            newnaxis_i[j], true));
      }
   }

   if(data_offset_)
   {
      extension_.push_front(new FitsIntCard("GCOUNT  ", gcount_, true));
      extension_.push_front(new FitsIntCard("PCOUNT  ", pcount_, true));
   }
   else
      extension_.push_front(new FitsBoolCard("EXTEND  ", true, true));

   checkNaxis(); // check syntax and reset data_length and naxis_array
}

int* FitsOut::region2Naxis(const util::Region& region, int& newnaxis) const
{
   const int dimension = region.getDim();
   newnaxis = dimension;
   int* const _newnaxis = new int [dimension];
   for(int i = 0; i < newnaxis; ++i)
   {
      _newnaxis[i] = region.getLength(i+1);
//       if(_newnaxis[i] == 1)
//       {
//          --newnaxis;
//          --i;
//       }
   }
   return _newnaxis;
}

void FitsOut::setGeometry(const int newbitpix, const util::Region& region, const bool append)
{
   int newnaxis;
   int* const _newnaxis = region2Naxis(region, newnaxis); 
   const int fd = setGeometry(newbitpix, newnaxis, _newnaxis, append);
   close(fd);
   bitpixout_ = newbitpix;
   delete [] _newnaxis;
}

// for extensions:
// add startoffset here and make data_offset_ point to right place
// change from "write" to memcpy for header
// startoffset = 0 resets file
// or: add (..., const bool append = false)
// change open call (O_TRUNC) only if appen==false
// read file length for startoffset <- start value of data_offset
// rest can remain almost as it is
// shift reset of keys after open / startoffset <- set SIMPLE / XTENSION in resetMand accordingly
int FitsOut::setGeometry(const int newbitpix,
                         const int newnaxis,
                         const int* newnaxis_i,
                         const bool append)
{
   bool	inmem = filename_.empty();
   int fd = 0;

   if (!inmem)
   {
	   int oflag = O_RDWR;
	   if(append) oflag |= O_APPEND;
	   else oflag |= O_CREAT|O_TRUNC;
	   // open file
	   fd = open(filename_.c_str(), oflag,
						   S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
	   if(fd < 0)
		  throw FitsException("cannot open file '" +
							  filename_ + "' for writing");
	   struct stat fdstat;
	   fstat(fd, &fdstat);
	   data_offset_ = fdstat.st_size;
   }
   else
   {
	   if (append)
		   data_offset_ = fitsinmemlength_;
	   else
		   data_offset_ = 0;
   }
   
   // reset mandatories
   resetMandatories(newbitpix, newnaxis, newnaxis_i);
   // reset array keywords
   resetArrayKeys();
   // write header
   const string headerstring = writeHeader(!ignorejunk_);
   data_offset_ += headerstring.length(); // reset offset to data segment

   if (!inmem)
   {
	   if( write(fd, headerstring.c_str(), headerstring.length() ) < 0 )
	   {
		  close(fd);
		  throw FitsException("cannot write to open file '" +
							  filename_ + "'");
	   }
	   // set length of file
	   if(data_length_ > 0)
	   {
		  if( ftruncate(fd, data_offset_ + off_t(PH_C.RECORD_LENGTH) *
						(( (data_length_ - 1) /
						   off_t(PH_C.RECORD_LENGTH)) + 1)
				       ) < 0)
		  {
			 close(fd);
			 throw FitsException("cannot (re)set file size for file '" +
								 filename_ + "'");
		  }
	   }
	   if(fsync(fd) < 0)
	   {
		  close(fd);
		  throw FitsException("cannot sync file '" +
							  filename_ + "' for writing");
	   }
   }
   else
   {
	   // set length of file
	   checkinmemalloc(0,
			           data_offset_ + off_t(PH_C.RECORD_LENGTH) *
			   			              (( (data_length_ - 1) /
			   						   off_t(PH_C.RECORD_LENGTH)) + 1));
	   memcpy(fitsinmemptr_, headerstring.c_str(), headerstring.length());
   }
   return fd;
}

void FitsOut::checkinmemalloc(const off_t write_offset, const int bytestobewritten)
{
   size_t	allocsizeneeded = write_offset + bytestobewritten;

   if (allocsizeneeded > *ptrfitsinmemlength_)
   {
	   unsigned char* newinmemptr = new unsigned char[allocsizeneeded];
	   if (fitsinmemptr_)
	   {
		   memcpy(newinmemptr, fitsinmemptr_, fitsinmemlength_);
		   delete fitsinmemptr_;
	   }
	   *ptrfitsinmemptr_ = fitsinmemptr_ = newinmemptr;
	   *ptrfitsinmemlength_ = fitsinmemlength_ = allocsizeneeded;
   }
}

void FitsOut::setRegion(const util::Region& fregion) throw(FitsException)
{
   // free old map/region (if exists)
   closeData();
   // set new region
   testRegion(fregion);
   regionptr_ = new util::Region(fregion);

   if(naxis_ == 0)
      throw FitsException("No data segment associated with header (NAXIS = 0).");
   
   const off_t bpix = off_t(bytpix_);
   const off_t doff = data_offset_;
   const off_t psize = off_t(getpagesize());

   off_t regstartoff = 0;
   off_t reglen      = 0;
   
   if(regionptr_ != NULL)
   {
      off_t regendoff   = 0;
      for(int naxiscounter = naxis_; naxiscounter > 1;)
      {
         regstartoff += off_t(regionptr_->getStart(naxiscounter)) - 1;
         regendoff += off_t(regionptr_->getEnd(naxiscounter)) - 1;
         --naxiscounter;
         const off_t nx_i = off_t(getNaxis(naxiscounter));
         regstartoff *= nx_i;
         regendoff *= nx_i;
      }
      regstartoff += off_t(regionptr_->getStart(1)) - 1;
      regendoff += off_t(regionptr_->getEnd(1)) - 1;
      regstartoff *= bpix;
      regendoff *= bpix;
      reglen = regendoff + bpix - regstartoff;
   }
   else
   {
      reglen = data_length_;
   }
   
   const size_t bytelength = ((doff + regstartoff) % psize) + reglen;
   const off_t fileoff = ((doff + regstartoff) / psize) * psize;

   bool	inMem = filename_.empty();

   if (!inMem)
   {
	   const int fd = open(filename_.c_str(), O_RDWR,
						   S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
	   if(fd < 0)
		  throw FitsException("cannot open file '" +
							  filename_ + "' for writing");

	   fitsdataptr_ = (unsigned char *) mmap(NULL, bytelength,
											 PROT_READ|PROT_WRITE, MAP_SHARED,
											 fd, fileoff);

	   close(fd);
   }
   else
   {
	   checkinmemalloc(fileoff, bytelength);
	   fitsdataptr_ = fitsinmemptr_ + fileoff;
   }
   if(fitsdataptr_ == (unsigned char *) MAP_FAILED)
   {
      throw FitsException("cannot map file for writing");
   }
   fitsmaplength_ = bytelength;
   fitsregionoffset_ = fileoff;
}

void FitsOut::openData(const int newbitpix,
                       const util::Region& region, const bool append) throw(FitsException)
{
   int newnaxis;
   int* const _newnaxis = region2Naxis(region, newnaxis); 
   openData(newbitpix, newnaxis, _newnaxis, append);
   delete [] _newnaxis;
}


void FitsOut::openData(const int newbitpix,
                       const int newnaxis,
                       const int* newnaxis_i, const bool append) throw(FitsException)
{
   // open file
   const int fd = setGeometry(newbitpix, newnaxis, newnaxis_i, append);

   // set length of file
   if(data_length_ > 0)
   {
	  const off_t psize = off_t(getpagesize());
	  const size_t bytelength = (data_offset_ % psize) + data_length_;
	  const off_t fileoff = (data_offset_ / psize) * psize;

	  bool	inmem = filename_.empty();

	  if (!inmem)
	  {
		 // map file
		 fitsdataptr_ = (unsigned char *) mmap(NULL, bytelength,
											   PROT_READ|PROT_WRITE, MAP_SHARED,
											   fd, fileoff);
		 if(fitsdataptr_ == (unsigned char *) MAP_FAILED)
		 {
		    close(fd);
			fitsdataptr_ = NULL;
			throw FitsException("cannot map file for writing");
		 }
		 fitsmaplength_ = bytelength;
      }
      else
      {
   	     checkinmemalloc(fileoff, bytelength);
	     fitsdataptr_ = fitsinmemptr_ + fileoff;
      }
      fitsregionoffset_ = fileoff;
	  resetPosition();
   }
   if (fd)
	   close(fd);
}

void FitsOut::closeData()
{
   if(regionptr_)
   {
      delete regionptr_;
      regionptr_ = NULL;
   }
   if(fitsdataptr_)
   {
      if (!filename_.empty())
    	  munmap((char *)fitsdataptr_, fitsmaplength_);
      fitsdataptr_ = NULL;
   }
   fitsstreamptr_ = NULL;
   fitsmaplength_ = 0;
}

void FitsOut::copyData(FitsIn& infile, const bool append)
{
   setBscale(infile.getBscale());
   setBzero(infile.getBzero());
   if(infile.getRegionLength() == 0)
   {
      openData(infile.bitpix_, infile.naxis_, infile.naxis_array_, append);
      if(infile.data_length_ > 0)
         copyRawData(begin(), infile.begin(), infile.data_length_);
   }
   else
   {
      int dimension = infile.naxis_;
      int* dim_ = new int[infile.naxis_];
      for(int i = 0; i < dimension; ++i)
      {
         dim_[i] = infile.getRegionLength(i+1);
         if(dim_[i] == 1)
         {
            --dimension;
            --i;
         }
      }
      openData(infile.bitpix_, dimension, dim_, append);
      delete [] dim_;
      copyRawRegionArray(begin(), infile.begin(), infile.bitpix_,
                         infile.getFullRegion(), infile.getRegion());
   }
   closeData();
}

unsigned char* FitsOut::begin() const
{
   if(fitsdataptr_ == NULL)
      throw FitsException("Attempt to write file of unspecified geometry.");
   return fitsdataptr_ + (data_offset_ - fitsregionoffset_);
}

unsigned char* FitsOut::end() const
{
   return (begin() + data_length_);
}

void FitsOut::resetPosition()
{
   fitsstreamptr_ = begin();
}

void FitsOut::setPosition(off_t offset)
{
   fitsstreamptr_ = begin() + offset;
}

ptrdiff_t FitsOut::getPosition() const
{
   return (streampos() - begin());
}

void FitsOut::setFilename(const string& path)
{
   filename_ = path;
}

void FitsOut::setBitpixOut(const int bpo) throw(FitsException)
{
   if(fitsdataptr_ == NULL)
   {
      bitpixout_ = bpo;
   }
   else
      throw FitsException("cannot reset bitpix while file is open");
}

void FitsOut::setOrigin(const string& orig)
{
   origin_ = orig;
   addValueCard("ORIGIN  ", orig, "FITS file originator");
}

int FitsOut::getBitpixOut() const
{
   return bitpixout_;
}

string FitsOut::getOrigin() const
{
   return origin_;
}


size_t FitsExtension::getExtNo() const
{
   return extno_;
}

bool FitsExtension::getPrimary() const
{
   return primary_;
}

void FitsExtension::resetExtNo()
{
   primary_ = true;
   extno_ = 0;
}

void FitsExtension::incExtNo()
{
   primary_ = false;
   ++extno_;
}

void FitsExtension::unsetPrimary()
{
   primary_ = false;
}


FitsExtensionIn::FitsExtensionIn(const char * path,
                                 const bool quiet_please,
                                 const bool ignore_header) :
   FitsIn(path, quiet_please, ignore_header), FitsExtension() 
{ }

FitsExtensionIn::FitsExtensionIn(const string & path,
                                 const bool quiet_please,
                                 const bool ignore_header) :
   FitsIn(path, quiet_please, ignore_header), FitsExtension() 
{ }

FitsExtensionIn::~FitsExtensionIn()
{
   if(extension_.size() > 0)
      extension_.clear();
}

// return FitsIn object refering to the first Extension which matches
// XTENSION, EXTNAME, and optionally EXTVER and and EXTLEVEL
FitsIn FitsExtensionIn::getExtension(const string& xtension, const string& extname,
                                     const int extver, const int extlevel)
{
   // logic here is: first catch without exception breaks the endless loop and returns
   // key/value missmatches throw but are caught within the loop -> onto the next extension
   // if no extension is left exception is thrown by getNextExtension() (actually by FitsHeader::readHeader())
   while(true)
   {
      FitsIn extin_i( getNextExtension() );
      try
      {
         if(extin_i.getString("XTENSION") != xtension || extin_i.getString("EXTNAME ") != extname)
            throw FitsException("XTENSION and/or EXTNAME missmatch");
         if(extver){
            if(extin_i.getInt("EXTVER  ") != extver)
               throw FitsException("EXTVER missmatch");
            if(extlevel)
               if(extin_i.getInt("EXTLEVEL") != extlevel)
                  throw FitsException("EXTVER missmatch");
         }
         return extin_i;
      }
      catch(FitsException e){}
   }
}

// return FitsIn object refering to Extension number extno
// set internal extno_ counter to extno
// if not already done fill internal vector with references to all Extensions up to extno
FitsIn FitsExtensionIn::getExtension(const size_t extno)
{
   if(extno == 0)
      throw FitsException("Use normal FitsIn methods to read primary HDU and data segment.");
   if( extension_.size() < extno )
   {
      if(extension_.size() == 0)
         extension_.push_back( FitsIn::getNextExtension() );
      size_t last_ext = extension_.size();
      while(last_ext < extno)
      {
         extension_.push_back(
            (extension_[last_ext - 1]).getNextExtension() );
         last_ext = extension_.size();
      }
   }
   extno_ = extno;
   return extension_[extno-1];
}

// return FitsIn object refering to "next" extension, increment internal extno_ counter
FitsIn FitsExtensionIn::getNextExtension()
{
   return getExtension(extno_ + 1);
}

FitsExtensionOut::FitsExtensionOut(const char* path) :
   FitsOut(path), FitsExtension()
{ }
FitsExtensionOut::FitsExtensionOut(const string& path) :
   FitsOut(path), FitsExtension()
{ }
FitsExtensionOut::FitsExtensionOut(const string& path, const FitsHeader& header,
                                   const bool quiet_please, const bool ign_junk) :
   FitsOut(path, header, quiet_please, ign_junk), FitsExtension()
{ }
FitsExtensionOut::FitsExtensionOut(const char* path, const FitsHeader& header,
                                   const bool quiet_please, const bool ign_junk) :
   FitsOut(path, header, quiet_please, ign_junk), FitsExtension()
{ }

FitsExtensionOut::~FitsExtensionOut()
{ }

const emptyData_ emptyData;

}

