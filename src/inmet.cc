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

#include <string.h>

#include "utils.hh"
#include "main_functions.hh"

#define Modules "azi_inc, poly_fit, poly_eval"

int main(int argc, char **argv)
{
    if (main_check_narg(argc, Modules))
        return EARG;
    
    if (ut_module_select("azi_inc")) {
        return azi_inc(argc, argv);
    }
    
    //else if (ut_module_select("poly_fit")) {
        //return poly_fit(argc, argv);
    //}

    //else if (ut_module_select("poly_eval")) {
        //return poly_eval(argc, argv);
    //}

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return EARG;
    }
}
