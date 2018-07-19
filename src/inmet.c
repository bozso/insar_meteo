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

#include <stdio.h>
#include <string.h>
#include "main_functions.h"

#define Modules "azi_inc, fit_orbit, eval_orbit"

int main(int argc, char **argv)
{
    if (argc < 2) {
        errorln("At least one argument, the module name, is required.\
                 \nModules to choose from: %s.", Modules);
        printf("Use --help or -h as the first argument to print the help message.\n");
        return err_arg;
    }
    
    if (module_select("azi_inc") || module_select("AZI_INC"))
        return azi_inc(argc, argv);
    
    else if (module_select("fit_orbit") || module_select("FIT_ORBIT"))
        return fit_orbit(argc, argv);

    else if (module_select("eval_orbit") || module_select("EVAL_ORBIT"))
        return eval_orbit(argc, argv);

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return err_arg;
    }
    
    return 0;
}
