/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: type_name.h 491 2011-09-02 19:36:39Z drory $
 * ---------------------------------------------------------------------
 *
 * Copyright (C)  Niv Drory <drory@mpe.mpg.de>
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


#ifndef __LTL_TYPE_NAME__
#define __LTL_TYPE_NAME__

#include <ltl/config.h>
#include <string>

#ifdef LTL_COMPLEX_MATH
#  include <complex>
#endif

namespace ltl {

/*! \file type_name.h
  Type traits to map from contained types of \c MArray, \c FVector, and \c FMatrix
  to type names.
*/

//@{
/*! trait classes to map from contained type to type name in string form
 */
template<typename T>
struct __ltl_type_name
{
      static std::string str() { return "T"; }
};

#define LTL_TYPE_NAME_TRAIT(T,TSTR) \
      template<>                    \
      struct __ltl_type_name<T >    \
      {                             \
            static std::string str(void) { return TSTR; } \
      }
LTL_TYPE_NAME_TRAIT(char,"char");
LTL_TYPE_NAME_TRAIT(short,"short");
LTL_TYPE_NAME_TRAIT(int,"int");
LTL_TYPE_NAME_TRAIT(long,"long");
LTL_TYPE_NAME_TRAIT(unsigned char,"unsigned char");
LTL_TYPE_NAME_TRAIT(unsigned short,"unsigned short");
LTL_TYPE_NAME_TRAIT(unsigned int,"unsigned int");
LTL_TYPE_NAME_TRAIT(unsigned long,"unsigned long");
LTL_TYPE_NAME_TRAIT(float,"float");
LTL_TYPE_NAME_TRAIT(double,"double");
#ifdef HAVE_LONG_DOUBLE
    LTL_TYPE_NAME_TRAIT(long double,"long double");
#endif
#ifdef LTL_COMPLEX_MATH
    LTL_TYPE_NAME_TRAIT(std::complex<float>,"complex float");
    LTL_TYPE_NAME_TRAIT(std::complex<double>,"complex double");
#  ifdef HAVE_LONG_DOUBLE
    LTL_TYPE_NAME_TRAIT(std::complex<long double>,"complex long double");
#  endif
#endif

#  define LTL_TYPE_NAME(T)  __ltl_type_name<T>::str()
//@}


}

#endif // __LTL_TYPE_NAME__

