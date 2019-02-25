/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: timer.h 491 2011-09-02 19:36:39Z drory $
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


#ifndef __UTIL_TIMER_H__
#define __UTIL_TIMER_H__

#include <ltl/config.h>

#ifdef TM_IN_SYS_TIME
#  include <sys/time.h>
#else
#  include <ctime>
#endif

#ifdef HAVE_GETRUSAGE
#  include <sys/resource.h>
#endif

namespace util {

//
// simple timer class
//

class Timer
{

   public:
      Timer()
            : t1_(0), t2_(0)
      { }

      void start()
      {
         t1_ = systemTime();
      }

      void stop()
      {
         t2_ = systemTime();
      }

      long double elapsedSeconds()
      {
         return t2_ - t1_;
      }

   private:
      long double systemTime()
      {
#ifdef HAVE_GETRUSAGE
         getrusage(RUSAGE_SELF, &resourceUsage_);
         double seconds = resourceUsage_.ru_utime.tv_sec
                          + resourceUsage_.ru_stime.tv_sec;
         double micros  = resourceUsage_.ru_utime.tv_usec
                          + resourceUsage_.ru_stime.tv_usec;
         return seconds + micros/1.0e6;
#else
         return clock() / (long double) CLOCKS_PER_SEC;
#endif
      }

#ifdef HAVE_GETRUSAGE
      struct rusage resourceUsage_;
#endif

      long double t1_, t2_;
};

}

#endif // __UTIL_TIMER_H__
