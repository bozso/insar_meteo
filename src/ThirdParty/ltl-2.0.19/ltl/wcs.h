/* -*- C++ -*-
 *
 * ---------------------------------------------------------------------
 * $Id: wcs.h 491 2011-09-02 19:36:39Z drory $
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

#ifndef __WCS_H__
#define __WCS_H__

#include <ltl/config.h>
#include <ltl/fmatrix.h>
#include <ltl/fmatrix/gaussj.h>
#include <ltl/io/fits_header.h>
#include <ltl/marray/simplevec.h>

namespace ltl {

using std::cos;
using std::sin;
using std::tan;
using std::asin;
using std::atan;
using std::atan2;
using std::fmod;
using std::fabs;
using ::hypot;

/*!
  Angles in WCS are deg due to FITS standard requirements.
  Within CCS angles must be transposed to rad because of libmath.
  External return methods should deliver again deg.
  Sorry for the inconvenience, not my fault.

*/

//! Base class for intermediate WCs calculation
template<int N>
class WCS
{

   public:
//       //! just apply CRPIXj, assume PCi_j and CDELTi to be unity
//       WCS(const FVector<double, N>& crpix_in) :
//          crpix_(crpix_in), pc_(0.0), cdelt_(1.0)
//       {
//          pc_.traceVector() = 1.0;
//       };
//       //! just apply CRPIXj and CDELTi, assume PCi_j to be unity
//       WCS(const FVector<double, N>& crpix_in,
//           const FVector<double, N>& cdelt_in) :
//          crpix_(crpix_in), pc_(0.0), cdelt_(cdelt_in)
//       {
//          pc_.traceVector() = 1.0;
//       };
//       //! just apply CRPIXj and CDi_j (STScI / Iraf syntax)
//       WCS(const FVector<double, N>& crpix_in,
//           const FMatrix<double, N, N>& cd_in) :
//          crpix_(crpix_in), pc_(cd_in), cdelt_(1.0)
//       { };
      //! full scheme
      WCS(const FVector<double, N>& crpix_in,
          const FMatrix<double, N, N>& pc_in,
          const FVector<double, N>& cdelt_in) :
         crpix_(crpix_in), pc_(pc_in), cdelt_(cdelt_in)
      { }

      //! return intermediate WCs
      inline FVector<double, N> x_(const FVector<double, N>& p) const
      {
//          cerr.precision(16);
//          cerr << "p: " << p << endl;
         return cdelt_ * dot(pc_, p - crpix_);
      }

      //! calculate orig. coordinates from intermed WCs (slow version!)
      inline FVector<double, N> p_(const FVector<double, N>& x) const
      {
//          cerr.precision(16);
//          cerr << "x: " << x << endl;
         return crpix_ + GaussJ<double, N>::solve(pc_, x / cdelt_);
      }

   protected:
      FVector<double, N> crpix_;
      FMatrix<double, N, N> pc_;
      FVector<double, N> cdelt_;

      inline static double deg2rad(const double deg)
      { return deg * (M_PI / 180.0); }
      inline static double rad2deg(const double rad)
      { return rad * (180.0 / M_PI); }
      inline static double cosdeg(double deg)
      {
         deg = fabs( fmod(deg, 360.0) );
         if(deg == 0.0) return 1.0;
         if(deg == 180.0) return -1.0;
         if( (deg == 90.0) || (deg == 270.0) ) return 0.0;
         return cos( deg2rad(deg) );
      }
      inline static double sindeg(double deg)
      {
         deg = fmod(deg, 360.0);
         if( (deg == 0.0) || (deg == 180.0) || (deg == -180.0) ) return 0.0;
         if( (deg == 90.0) || (deg == -270.0) ) return 1.0;
         if( (deg == -90.0) || (deg == 270.0) ) return -1.0;
         return sin( deg2rad(deg) );
      }
};

//! Base CCS class, use to derive the different projection / CCS
/*!

*/
class CCS : public WCS<2>
{
   public:
      //! full scheme
      CCS(const FVector<double, 2>& crpix_in,
          const FMatrix<double, 2, 2>& pc_in,
          const FVector<double, 2>& cdelt_in,
          const double phi_0in, const double theta_0in,
          const double alpha_0in, const double delta_0in,
          const double alpha_pin, const double delta_pin) :
         WCS<2>(crpix_in, pc_in, cdelt_in),
         phi_0(deg2rad(phi_0in)), theta_0(deg2rad(theta_0in)),
         alpha_0(deg2rad(alpha_0in)), delta_0(deg2rad(delta_0in)),
         phi_p( ( (delta_0 >= theta_0) ? 0.0 : M_PI ) ), // LONPOLE
         theta_p( M_PI_2 ), // LATPOLE
         alpha_p(deg2rad(alpha_pin)), delta_p(deg2rad(delta_pin)),
         c_dp( cosdeg( delta_pin ) ), s_dp( sindeg( delta_pin ) )
      { }

      virtual ~CCS()
      { }

      virtual FVector<double, 2> x_nc(const FVector<double, 2>& nc) const = 0;

      //! calculate native coordinates \f$\theta\f$, \f$\phi\f$ in rad from pixel coordinates
      virtual FVector<double, 2> nc_(const FVector<double, 2>& p) const = 0;

      //! calculate native coordinates \f$\theta\f$, \f$\phi\f$ in rad from celestial coordinates
      inline FVector<double, 2> nc_cc(const FVector<double, 2>& cc) const
      {
         const double alpha = cc(1);
         const double ama   = alpha - alpha_p;
         const double s_ama = sin(ama);
         const double c_ama = cos(ama);
         const double delta = cc(2);
         const double s_d   = sin(delta);
         const double c_d   = cos(delta);
         FVector<double, 2> nc;
         nc(1) = phi_p +
            atan2( 0.0 - (c_d * s_ama),
                   (s_d * c_dp) - (c_d * s_dp * c_ama) );
         nc(2) = asin( (s_d * s_dp) + (c_d * c_dp * c_ama) );
//          cerr.precision(16);
//          cerr << "CC: " << cc << endl
//               << "NC: " << nc << endl;
         return nc;
      }

      //! calculate celestial coordinates \f$\alpha\f$, \f$\delta\f$ from native coords in rad
      inline FVector<double, 2> cc_(const FVector<double, 2>& nc) const
      {
         const double phi   = nc(1);
         const double pmp   = phi - phi_p;
         const double c_p   = cos(pmp);
         const double s_p   = sin(pmp);
         const double theta = nc(2);
         const double c_t   = cos(theta);
         const double s_t   = sin(theta);
         FVector<double, 2> cc;
         cc(1) = alpha_p +
            atan2( 0.0 - (c_t * s_p),
                   (s_t * c_dp) - (c_t * s_dp * c_p) );
         cc(2) = asin( (s_t * s_dp) + (c_t * c_dp * c_p) );
//          cerr.precision(16);
//          cerr << "NC: " << nc << endl
//               << "CC: " << cc << endl;
         return cc;
      }

      //! calculate projection \f$\alpha\f$, \f$\delta\f$ from given pixel(x, y) in deg
      virtual FVector<double, 2> solve(const FVector<double, 2>& p) const = 0;

      //! calculate pixel(x, y) in deg  from given projection \f$\alpha\f$, \f$\delta\f$
      virtual FVector<double, 2> solve_inv(const FVector<double, 2>& radec) const = 0;

   protected:

      //! angles in rad
      double phi_0, theta_0;
      double alpha_0, delta_0;
      double phi_p, theta_p;
      double alpha_p, delta_p;
      double c_dp, s_dp;
};

//! Gnonomic CCS, TAN representation
class CCS_TAN : public CCS
{
   public:
      CCS_TAN(const FVector<double, 2>& crpix_in,
              const FMatrix<double, 2, 2>& pc_in,
              const FVector<double, 2>& cdelt_in,
              const FVector<double, 2>& crval_in) :
         CCS(crpix_in, pc_in, cdelt_in,
             0.0, 90.0,
             crval_in(1), crval_in(2),
             crval_in(1), crval_in(2))
      { }
      
      virtual FVector<double, 2> x_nc(const FVector<double, 2>& nc) const
      {
         const double phi   = nc(1);
         const double theta = nc(2);
         const double r_t   = 1.0 / tan(theta);
         FVector<double, 2> x;
         x(1) = rad2deg(r_t * sin(phi));
         x(2) = rad2deg(-r_t * cos(phi));
//          cerr.precision(16);
//          cerr << "NC: " << nc << endl
//               << "x:  " << x << endl;
         return x;
      }
      
      //! calculate native coordinates \f$\theta\f$, \f$\phi\f$
      virtual FVector<double, 2> nc_(const FVector<double, 2>& p) const
      {
         // intermediate WCs in rad (x, y)
         const FVector<double, 2> x( x_(p) * M_PI / 180.0 );
         // native CCs in rad (phi, theta)
         FVector<double, 2> nc;
         nc(1) = atan2( x(1), -x(2) );
         nc(2) = atan( 1.0 / hypot(x(1), x(2)) );
//          cerr.precision(16);
//          cerr << "x:  " << x << endl
//               << "NC: " << nc << endl;
         return nc;
      }

      //! calculate projection \f$\alpha\f$, \f$\delta\f$ from given pixel(x, y) in deg
      virtual FVector<double, 2> solve(const FVector<double, 2>& p) const
      {
         return cc_( nc_(p) ) * (180.0 / M_PI);
      }

      //! calculate pixel(x, y) in deg  from given projection \f$\alpha\f$, \f$\delta\f$
      virtual FVector<double, 2> solve_inv(const FVector<double, 2>& radec) const
      {
         return p_( x_nc( nc_cc(radec * (M_PI / 180.0)) ) );
      }
};

//! CCS interface class.
/*!
  Holds conversion and return functions as well as FITS header parser.
*/
class FitsCCS
{
   public:
      FitsCCS(const FitsHeader& hdr);
      ~FitsCCS()
      { delete ccs; }
      //! Convert pixel to RA / Dec applying CCS
      FVector<double, 2> getRADEC(const FVector<double, 2>& p) const;
      inline FVector<double, 2> getRADEC(const FixedVector<int, 2>& p) const
      {
         FVector<double, 2> mp;
         mp(1) = p(1); mp(2) = p(2);
         return getRADEC(mp);
      }

      //! Convert RA / Dec to pixel applying CCS
      FVector<double, 2> radecToPixel(const FVector<double, 2>& cc) const;
      inline FVector<double, 2> radecToPixel(const FixedVector<int, 2>& cc) const
      {
         FVector<double, 2> mcc;
         mcc(1) = cc(1); mcc(2) = cc(2);
         return radecToPixel(mcc);
      }

      
   protected:
      CCS* ccs;
};

}

#endif
