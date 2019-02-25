/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: region.h 494 2011-11-04 14:22:48Z cag $
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


#ifndef __REGION_H__
#define __REGION_H__

#include <ltl/config.h>
#include <ltl/util/u_exception.h>

#include <string>

#include <cstdlib>
#include <cstdio>

using std::size_t;

namespace util {

//! To represent geometries or subarrays of N-dimensional arrays.
class Region
{
   private:
      //! Empty constructor is private!
      Region() 
         : N_(0), start_(NULL), end_(NULL)
      { }
   public:
      //! Construct an \e N dimensional Region.
      Region(const size_t N);
      //! Construct as copy from \e other.
      Region(const Region& other);
      //! Construct as \e N dimensional Region from input string with format start_1:end_1,...,start_N:end_N.
      Region(const string& s, const size_t N);
      ~Region()
      { delete [] start_; }

      //! Return number of dimensions.
      size_t getDim() const
      { return N_; }
      //! Return start coordinate of dimension \e i.
      int getStart(const size_t i) const throw(UException);
      //! Return end coordinate of dimension \e i.
      int getEnd(const size_t i) const throw(UException);
      //! Return size of Region.
      size_t getLength() const throw(UException);
      //! Return length along dimension \e i.
      size_t getLength(const size_t i) const throw(UException);
      //! Return a (rand reduced) slice of Region.
      Region getSlice(const size_t startdim,
                      const size_t enddim) const
         throw(UException);
      //! Set start coordinate of dimension \e i to \e x.
      void setStart(const size_t i, const int x) throw(UException);
      //! Set end coordinate of dimension \e i to \e x.
      void setEnd(const size_t i, const int x) throw(UException);
      //! Set range of dimension \e i to start at \e s and end at \e e.
      void setRange(const size_t i, const int s, const int e)
         throw(UException);
      //! Copy settings from \e region.
      void setRegion(const Region &region) throw(UException);
      //! Parse Region to a string with format start_1:end_1,...,start_N:end_N.
      string toString() const;

   protected:
      //! Number of dimensions.
      const size_t N_;
      //! Pointer to array with start coordinates.
      int *const start_;
      //! Pointer to array with end coordinates.
      int *const end_;

      //! Error message for invalid dimension requests.
      static const string error_get_dim;
      //! Error message for invalid dimension requests.
      static const string error_set_dim;
};

}

/***********************************************************
 * Doxygen documentation block starts here
 **********************************************************/

/*! \class util::Region
  A utility class to allow an easy selection of regions within
  N-dimensional arrays.
  Used for Interfacing with command line, config file and FITS I/O.
  \throw UException on request of illegal (i.e. > N or < 0) dimensions.
*/

#endif
