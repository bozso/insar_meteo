/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SATORBIT_H
#define SATORBIT_H

#include "utils.hh"
#include "eigen_aux.hh"

/***********
 * Structs *
 ***********/


// cartesian coordinate
struct cart {
    double x, y, z;
    
    cart() {};
    
    cart(double _x, double _y, double _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
};

void ell_cart (utils::cdouble lon, utils::cdouble lat, utils::cdouble h,
               double& x, double& y, double& z);

void cart_ell(utils::cdouble x, utils::cdouble y, utils::cdouble z,
              double& lon, double& lat, double& h);

void calc_azi_inc(const Eigen::poly_fit& orb, utils::cdouble X, utils::cdouble Y,
                  utils::cdouble Z, utils::cdouble lon, utils::cdouble lat,
                  utils::cuint max_iter, double& azi, double& inc);

#endif // SATORBIT_HPP
