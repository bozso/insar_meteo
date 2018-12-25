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


#include <tgmath.h>

#include "utils.h"
#include "satorbit.h"


/************************
 * Auxilliary functions *
 * **********************/


/***********************************************
 * Main functions - calleble from command line *
 ***********************************************/

int calc_azi_inc(int argc, char **argv)
{
    argp const ap = {argc, 5, argv, "calc_azi_inc"
    "[fit_file] [coords] [mode] [max_iter] [outfile]\
     \n \
     \n fit_file - (ascii, in) contains fitted orbit polynom parameters\
     \n coords   - (binary, in) inputfile with coordinates\
     \n mode     - xyz for WGS-84 coordinates, llh for WGS-84 lon., lat., height\
     \n max_iter - maximum number of iterations when calculating closest approache\
     \n outfile  - (binary, out) azi, inc pairs will be printed to this file\
     \n\n"};
    
    uint max_iter = 0;
    
    if (check_narg(&ap, 5) or get_arg(&ap, 4, "%u", &max_iter))
        return 1;
    
    File coords_file = open(coords_file, argv[3], "rb"),
         out_file = open(out_file, argv[6], "wb");
    
    if (not(coords_file and out_file))
        return 1;
    
    double coords[3], azi_inc[2];
    fit_poly orb;

    // topocentric parameters in PS local system
    double X, Y, Z,
           lon, lat, h,
           azi, inc;
    
    if (read_fit(&orb, argv[2])) {
        Perror("azi_inc", "Could not read orbit fit file %s. Exiting\n!", argv[2]);
        return 1;
    }
    
    size_t sdouble = sizeof(double);

    
    // infile contains lon, lat, h
    if (str_equal(argv[4], "llh")) {
        while (readb(coords_file, sdouble, 3, coords) > 0) {
            lon = coords[0] * DEG2RAD;
            lat = coords[1] * DEG2RAD;
            h   = coords[2];
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, X, Y, Z);
            
            _calc_azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                         _azi_inc[0], _azi_inc[1]);

            writeb(out_file, sdouble, 2, azi_inc);
        } // end while
    }
    // infile contains X, Y, Z
    else if (str_equal(argv[4], "xyz")) {
        while (readb(coords_file, sdouble, 3, coords) > 0) {
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, coords[0], coords[1], coords[2]);
            
            _calc_azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                         _azi_inc[0], _azi_inc[1]);

            write(out_file, sdouble, 2, azi_inc);
        } // end while
    } // end else if
    else {
        errorln("Third argument should be either llh or xyz not %s!",
                argv[4]);
        return 1;
    }

    return 0;
}
