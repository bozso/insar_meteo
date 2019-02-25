/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: fits_data.h 541 2014-07-09 17:01:12Z drory $
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


#ifndef __FITS_DATA_H__
#define __FITS_DATA_H__

#include <ltl/config.h>
#include <ltl/util/region.h>
#include <ltl/misc/swapbytes.h>
#include <ltl/misc/stdint_hack.h>

#include <cstdlib>
#include <cstring>

#define LTL_FITS_BITPIX_CHAR     8
#define LTL_FITS_BITPIX_SHORT   16
#define LTL_FITS_BITPIX_INT     32
#define LTL_FITS_BITPIX_FLOAT  -32
#define LTL_FITS_BITPIX_DOUBLE -64

//using util::Region; // bad idea to have "using" in a library
using util::UException;

namespace ltl {

// just copy data ignoring endianess
/*! \relates ltl::FitsIO
  Wrapper for memcpy which returns an "end iterator".
*/
inline uint8_t* copyRawData(uint8_t* destptr, const uint8_t* sourceptr, size_t len)
{
   memcpy(destptr, sourceptr, len);
   return destptr + len;
}

/*! \relates ltl::sData
  Increase data segment pointer according to BITPIX.
*/
template<int BITPIX>
inline void ppDataPointer( uint8_t* & /*dataptr*/ )
{ }

/*! \relates ltl::sData<T, 8>
  Increase data segment pointer according to BITPIX=8.
*/
template<>
inline void ppDataPointer<LTL_FITS_BITPIX_CHAR>( uint8_t* & dataptr )
{
   dataptr += sizeof(uint8_t);
}

/*! \relates ltl::sData<T, 16>
  Increase data segment pointer according to BITPIX=16.
*/
template<>
inline void ppDataPointer<LTL_FITS_BITPIX_SHORT>( uint8_t* & dataptr )
{
   dataptr += sizeof(int16_t);
}

/*! \relates ltl::sData<T, 32>
  Increase data segment pointer according to BITPIX=32.
*/
template<>
inline void ppDataPointer<LTL_FITS_BITPIX_INT>( uint8_t* & dataptr )
{
   dataptr += sizeof(int32_t);
}

/*! \relates ltl::sData<T, -32>
  Increase data segment pointer according to BITPIX=-32.
*/
template<>
inline void ppDataPointer<LTL_FITS_BITPIX_FLOAT>( uint8_t* & dataptr )
{
   dataptr += sizeof(float);
}

/*! \relates ltl::sData<T, -64>
  Increase data segment pointer according to BITPIX=-64.
*/
template<>
inline void ppDataPointer<LTL_FITS_BITPIX_DOUBLE>( uint8_t* & dataptr )
{
   dataptr += sizeof(double);
}

/*! 
  Read single data and return requested value type.
  Write single data to data segment.
  Specialisations for every FITS BITPIX.
*/
template<class T, int BITPIX>
struct sData
{ };

template<class T>
struct sData<T, LTL_FITS_BITPIX_CHAR>
{
      inline static T read( uint8_t* const sourceptr )
      {
         return T(*sourceptr);
      }
      inline static T read( uint8_t* const sourceptr, const double bscale, const double bzero)
      {
         return T( bscale * sData<double, LTL_FITS_BITPIX_CHAR>::read(sourceptr) + bzero);
      }
      inline static T read( uint8_t* const sourceptr, const int bscale, const int bzero)
      {
         return T( bscale * sData<int, LTL_FITS_BITPIX_CHAR>::read(sourceptr) + bzero);
      }

      inline static void write( uint8_t* const destptr, const T value )
      {
         *destptr = uint8_t(value);
      }
};

template<class T>
struct sData<T, LTL_FITS_BITPIX_SHORT>
{
      typedef union { int16_t i; uint16_t u; } U;

      inline static T read( uint8_t* const sourceptr )
      {
#ifdef WORDS_BIGENDIAN
         return T(* (int16_t *) sourceptr);
#else
         U u;
         u.u = * (uint16_t *)sourceptr;
         u.u = LTL_FROM_BIGENDIAN_16( u.u );
         return T(u.i);
#endif
      }
      inline static T read( uint8_t* const sourceptr, const double bscale, const double bzero)
      {
         return T( bscale * sData<double, LTL_FITS_BITPIX_SHORT>::read(sourceptr) + bzero);
      }

      inline static T read( uint8_t* const sourceptr, const int bscale, const int bzero)
      {
         return T( bscale * sData<int, LTL_FITS_BITPIX_SHORT>::read(sourceptr) + bzero);
      }

      inline static void write( uint8_t* const destptr, const T value )
      {
#ifdef WORDS_BIGENDIAN
         * (int16_t *) destptr = int16_t(value);
#else
         U u;
         u.i = int16_t(value);
         * (uint16_t *) destptr = LTL_TO_BIGENDIAN_16( u.u );
#endif
}

};

template<class T>
struct sData<T, LTL_FITS_BITPIX_INT>
{
      typedef union { int32_t i; uint32_t u; } U;

      inline static T read( uint8_t* const sourceptr )
      {
#ifdef WORDS_BIGENDIAN
         return T(* (int32_t *) sourceptr);
#else
         U u;
         u.u = * (uint32_t *)sourceptr;
         u.u = LTL_FROM_BIGENDIAN_32( u.u );
         return T(u.i);
#endif
      }
      inline static T read( uint8_t* const sourceptr, const double bscale, const double bzero)
      {
         return T( bscale * sData<double, LTL_FITS_BITPIX_INT>::read(sourceptr) + bzero);
      }
      inline static T read( uint8_t* const sourceptr, const int bscale, const int bzero)
      {
         return T( double(bscale) * sData<double, LTL_FITS_BITPIX_INT>::read(sourceptr) +
                   double(bzero) );
      }

      inline static void write( uint8_t* const destptr, const T value )
      {
#ifdef WORDS_BIGENDIAN
         * (int32_t *) destptr = int32_t(value);
#else
         U u;
         u.i = int32_t(value);
         * (uint32_t *) destptr = LTL_TO_BIGENDIAN_32( u.u );
#endif
      }
};


template<class T>
struct sData<T, LTL_FITS_BITPIX_FLOAT>
{
      typedef union { float f; uint32_t u; } U;

      inline static T read( uint8_t* const sourceptr )
      {
#ifdef WORDS_BIGENDIAN
         return T(* (float *) sourceptr);
#else
         U u;
         u.u = * (uint32_t *)sourceptr;
         u.u = LTL_FROM_BIGENDIAN_32( u.u );
         return T(u.f);
#endif
      }
      inline static T read( uint8_t* const sourceptr, const double bscale, const double bzero)
      {
         return T( bscale * sData<double, LTL_FITS_BITPIX_FLOAT>::read(sourceptr) + bzero);
      }
      inline static T read( uint8_t* const sourceptr, const int bscale, const int bzero)
      {
         return T( double(bscale) * sData<double, LTL_FITS_BITPIX_FLOAT>::read(sourceptr) +
                   double(bzero) );
      }

      inline static void write( uint8_t* const destptr, const T value )
      {
#ifdef WORDS_BIGENDIAN
         * (float *) destptr = float(value);
#else
         U u;
         u.f = float(value);
         * (uint32_t *) destptr = LTL_TO_BIGENDIAN_32( u.u );
#endif
      }
};


template<class T>
struct sData<T, LTL_FITS_BITPIX_DOUBLE>
{
      typedef union { double f; uint64_t u; } U;

      inline static T read( uint8_t* const sourceptr )
      {
#ifdef WORDS_BIGENDIAN
         return T(* (double *) sourceptr);
#else
         U u;
         u.u = * (uint64_t *)sourceptr;
         u.u = LTL_FROM_BIGENDIAN_64( u.u );
         return T(u.f);
#endif
      }
      inline static T read( uint8_t* const sourceptr, const double bscale, const double bzero)
      {
         return T( bscale * sData<double, LTL_FITS_BITPIX_DOUBLE>::read(sourceptr) + bzero);
      }
      inline static T read( uint8_t* const sourceptr, const int bscale, const int bzero)
      {
         return T( double(bscale) * sData<double, LTL_FITS_BITPIX_DOUBLE>::read(sourceptr) +
                   double(bzero) );
      }

      inline static void write( uint8_t* const destptr, const T value )
      {
#ifdef WORDS_BIGENDIAN
         * (double *) destptr = double(value);
#else
         U u;
         u.f = double(value);
         * (uint64_t *) destptr = LTL_TO_BIGENDIAN_64( u.u );
#endif
      }
};


/*! \relates ltl::FitsIn
  Read data from data segment into array of type T.
*/
template<class T, int BITPIX>
inline T* readData(T* destptr, uint8_t* sourceptr, const size_t bytlen)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *destptr = sData<T, BITPIX>::read(sourceptr);
      ++destptr;
      ppDataPointer<BITPIX>(sourceptr);
   }
   return destptr;
}

template<>
inline uint8_t* readData<uint8_t, LTL_FITS_BITPIX_CHAR>(uint8_t* destptr,
                                                        uint8_t* sourceptr,
                                                        const size_t bytlen)
{
   return copyRawData(destptr, sourceptr, bytlen);
}

#ifdef WORDS_BIGENDIAN
template<>
inline int16_t* readData<int16_t, LTL_FITS_BITPIX_SHORT>(int16_t* destptr,
                                                         uint8_t* sourceptr,
                                                         const size_t bytlen)
{
   uint8_t* bdestptr = (uint8_t*)destptr;
   return (int16_t*) copyRawData(bdestptr, sourceptr, bytlen);
}

template<>
inline int32_t* readData<int32_t, LTL_FITS_BITPIX_INT>(int32_t * destptr,
                                                       uint8_t* sourceptr,
                                                       const size_t bytlen)
{
   uint8_t* bdestptr = (uint8_t*)destptr;
   return (int32_t*) copyRawData(bdestptr, sourceptr, bytlen);
}

template<>
inline float* readData<float, LTL_FITS_BITPIX_FLOAT>(float* destptr,
                                                     uint8_t* sourceptr,
                                                     const size_t bytlen)
{
   uint8_t* bdestptr = (uint8_t*)destptr;
   return (float*) copyRawData(bdestptr, sourceptr, bytlen);
}

template<>
inline double* readData<double, LTL_FITS_BITPIX_DOUBLE>(double* destptr,
                                                        uint8_t* sourceptr,
                                                        const size_t bytlen)
{
   uint8_t* bdestptr = (uint8_t*)destptr;
   return (double*) copyRawData(bdestptr, sourceptr, bytlen);
}
#endif

/*! \relates ltl::FitsIn
  Read data from data segment into array of type T applying arbitrary 
  BSCALE and BZERO.
*/
template<class T, int BITPIX>
T* readData(T* destptr, uint8_t* sourceptr, const size_t bytlen,
              const double bscale, const double bzero)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *destptr = sData<T, BITPIX>::read(sourceptr, bscale, bzero);
      ++destptr;
      ppDataPointer<BITPIX>(sourceptr);
   }
   return destptr;
}

/*! \relates ltl::FitsIn
  Read data from data segment into array of type T applying integer
  BSCALE and BZERO.
*/
template<class T, int BITPIX>
T* readData(T* destptr, uint8_t* sourceptr, const size_t bytlen,
            const int bscale, const int bzero)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *destptr = sData<T, BITPIX>::read(sourceptr, bscale, bzero);
      ++destptr;
      ppDataPointer<BITPIX>(sourceptr);
   }
   return destptr;
}

/*! \relates ltl::FitsIn
  Read data from data segment into STL conform container.
*/
template<class T, int BITPIX>
void readIData(T& i,
               uint8_t* sourceptr, const size_t bytlen)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *i = sData<typename T::value_type, BITPIX>::read(sourceptr);
      ppDataPointer<BITPIX>(sourceptr);
      ++i;
   }
}

/*! \relates ltl::FitsIn
  Read data from data segment into STL conform container applying arbitrary
  BSCALE and BZERO.
*/
template<class T, int BITPIX>
void readIData(T& i,
               uint8_t* sourceptr, const size_t bytlen,
               const double bscale, const double bzero)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *i = sData<typename T::value_type, BITPIX>::read(sourceptr, bscale, bzero);
      ppDataPointer<BITPIX>(sourceptr);
      ++i;
   }
}

/*! \relates ltl::FitsIn
  Read data from data segment into STL conform container applying integer
  BSCALE and BZERO.
*/
template<class T, int BITPIX>
void readIData(T& i,
               uint8_t* sourceptr, const size_t bytlen,
               const int bscale, const int bzero)
{
   uint8_t* const endptr = sourceptr + bytlen;
   while(sourceptr < endptr)
   {
      *i = sData<typename T::value_type, BITPIX>::read(sourceptr, bscale, bzero);
      ppDataPointer<BITPIX>(sourceptr);
      ++i;
   }
}

/*! \relates ltl::FitsOut
  Write data from STL conform container into data segment.
*/
template<class T, int BITPIX>
void writeIData(uint8_t* destptr, T& i,
                const size_t bytlen)
{
   uint8_t* const endptr = destptr + bytlen;
   while(destptr < endptr)
   {
      sData<typename T::value_type, BITPIX>::write(destptr, *i);
      ppDataPointer<BITPIX>(destptr);
      ++i;
   }
}

/*! \relates ltl::FitsIn
  Return an array of type T and size \e bytlen (in bytes)
  holding data of data segment.
*/
template<class T, int BITPIX>
T * getDataArray( uint8_t* sourceptr, const size_t bytlen,
                  const double bscale, const double bzero)
{
   const size_t bytpix = size_t(::abs(BITPIX)) / size_t(8);
   const size_t npixel = bytlen / bytpix;
   T * arrayptr = new T [npixel];

   if( (bscale == 1.0) && (bzero == 0.0) )
      readData<T, BITPIX>(arrayptr, sourceptr, bytlen);
   else
   {
      const int ibscale = int(bscale);
      const int ibzero  = int(bzero);
      if( (double(ibscale) == bscale) && (double(ibzero) == bzero) )
         readData<T, BITPIX>(arrayptr, sourceptr, bytlen, ibscale, ibzero);
      else
         readData<T, BITPIX>(arrayptr, sourceptr, bytlen, bscale, bzero);
   }
   return arrayptr;
}

/*! \relates ltl::FitsIn
  Read data of data segment into container via iterator of type T.
*/
template<class T, int BITPIX>
void readDataArray(T i, uint8_t* sourceptr, const size_t bytlen,
                   const double bscale, const double bzero)
{
   if( (bscale == 1.0) && (bzero == 0.0) )
      readIData<T, BITPIX>(i, sourceptr, bytlen);
   else
   {
      const int ibscale = int(bscale);
      const int ibzero  = int(bzero);
      if( (double(ibscale) == bscale) && (double(ibzero) == bzero) )
         readIData<T, BITPIX>(i, sourceptr, bytlen, ibscale, ibzero);
      else
         readIData<T, BITPIX>(i, sourceptr, bytlen, bscale, bzero);
   }
}


/*! \relates ltl::FitsIn
  Return a raw subarray of type T of data segment.
*/
uint8_t* getRawRegionArray(uint8_t* const sourceptr, const int bitpix,
                           const util::Region& fullframe, const util::Region& region);

/*! \relates ltl::FitsIn
  Copy raw subarray of data segment.
*/
void copyRawRegionArray(uint8_t* const regionptr, uint8_t* const sourceptr,
                        const int bitpix,
                        const util::Region& fullframe, const util::Region& region);

/*! \relates ltl::FitsIn
  Return a subarray of data segment translated into type T.
*/
template<class T, int BITPIX>
T* getRegionArray(uint8_t* const sourceptr,
                  const util::Region& fullframe, const util::Region& region,
                  const double bscale, const double bzero)
{
   const size_t bytpix = size_t(::abs(BITPIX / 8));
   const bool nobany = ((bscale == 1.0) && (bzero == 0.0));
   const bool bint   = ( (bscale == double(int(bscale))) &&
                         (bzero == double(int(bzero))) );

   const size_t naxis = fullframe.getDim();
   size_t* const counterarray = new size_t[naxis];
   size_t* const regioncounter = counterarray - 1;
   for(size_t naxiscounter = 1; naxiscounter <= naxis; ++naxiscounter)
      regioncounter[naxiscounter] = 0;

   // calculate new data length in type T
   const size_t length = region.getLength();
   // allocate data array
   T* const regionptr = new T [length];
   T* const destend = regionptr + length;
   T* destptr = regionptr;
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
      
      if( nobany )
         destptr = readData<T, BITPIX>(destptr, sourceptr + srcoffset, bytlinlength);
      else
      {
         if(bint)
            destptr = readData<T, BITPIX>(destptr,
                                          sourceptr + srcoffset, bytlinlength,
                                          int(bscale), int(bzero));
         else
            destptr = readData<T, BITPIX>(destptr,
                                          sourceptr + srcoffset, bytlinlength,
                                          bscale, bzero);
      }
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
   return regionptr;
}

/*! \relates ltl::FitsIn
  Read \e region of data segment into STL conform container
  translating to type T.
*/
template<class T, int BITPIX>
void readRegionArray(T i, uint8_t* const sourceptr,
                     const util::Region& fullframe, const util::Region& region,
                     const double bscale, const double bzero)
{
   const size_t bytpix = size_t(::abs(BITPIX / 8));
   const bool nobany = ((bscale == 1.0) && (bzero == 0.0));
   const bool bint   = ( (bscale == double(int(bscale))) &&
                         (bzero == double(int(bzero))) );

   const size_t naxis = fullframe.getDim();
   size_t* const counterarray = new size_t[naxis];
   size_t* const regioncounter = counterarray - 1;
   for(size_t naxiscounter = 1; naxiscounter <= naxis; ++naxiscounter)
      regioncounter[naxiscounter] = 0;

   // calculate new data length in bytes
   const size_t length = region.getLength();
   const size_t destlinlength = region.getLength(1);
   const size_t bytlinlength = bytpix * destlinlength;
   size_t destoffset = 0;
   //cout<<"bytpix: "<<bytpix<<", length "<<length<<", naxis "<<naxis<<"\n";
   while(destoffset < length)
   { // copy until all done
      size_t srcoffset=0;
      for(size_t naxiscounter = naxis; naxiscounter > 1;)
      {
         srcoffset += region.getStart(naxiscounter) + regioncounter[naxiscounter] - 1;
         srcoffset *= fullframe.getLength(--naxiscounter);
      }
      srcoffset += region.getStart(1) + regioncounter[1] - 1;
      srcoffset *= bytpix;
      
      if( nobany )
         readIData<T, BITPIX>(i, sourceptr + srcoffset, bytlinlength);
      else
      {
         if(bint)
            readIData<T, BITPIX>(i, sourceptr + srcoffset, bytlinlength,
                                 int(bscale), int(bzero));
         else
            readIData<T, BITPIX>(i, sourceptr + srcoffset, bytlinlength,
                                 bscale, bzero);
      }
      destoffset += destlinlength; // calculate next destinationoffset
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


/*! \relates ltl::FitsOut
  Write from STL-conform container into data segment.
*/
template<class T, int BITPIX>
void writeData(uint8_t* destptr,
               const size_t bytlen,
               T i)
{
   uint8_t* const endptr = destptr + bytlen;
   while(destptr < endptr)
   {
      sData<typename T::value_type, BITPIX>::write(destptr, *i);
      ppDataPointer<BITPIX>(destptr);
      ++i;
   }
}

/*! \relates ltl::FitsOut
  Write from STL conform container into \e region of data segment
  translating to BITPIX.
*/
template<class T, int BITPIX>
void writeRegionArray(uint8_t* const destptr, T i,
                      const util::Region& fullframe, const util::Region& region)
{
   const size_t bytpix = size_t(::abs(BITPIX / 8));
   const size_t naxis = fullframe.getDim();
   size_t* const counterarray = new size_t[naxis];
   size_t* const regioncounter = counterarray - 1;
   for(size_t naxiscounter = 1; naxiscounter <= naxis; ++naxiscounter)
      regioncounter[naxiscounter] = 0;

   // calculate data length in bytes
   const size_t length = region.getLength();
   const size_t srclinlength = region.getLength(1);
   const size_t bytlinlength = bytpix * srclinlength;
   size_t srcoffset=0;
   //cout<<"bytpix: "<<bytpix<<", length "<<length<<", naxis "<<naxis<<"\n";
   while(srcoffset < length)
   { // copy until all done
      size_t destoffset = 0;
      for(size_t naxiscounter = naxis; naxiscounter > 1;)
      {
         destoffset += region.getStart(naxiscounter) + regioncounter[naxiscounter] - 1;
         destoffset *= fullframe.getLength(--naxiscounter);
      }
      destoffset += region.getStart(1) + regioncounter[1] - 1;
      destoffset *= bytpix;
      writeIData<T, BITPIX>(destptr + destoffset, i, bytlinlength);
      srcoffset += srclinlength; // calculate next destinationoffset
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

#endif
