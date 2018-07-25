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

#include <iostream>
#include "main_functions.hpp"

#define Modules "azi_inc, fit_orbit, eval_orbit"

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2) {
        cerr << "At least one argument, the module name, is required.\
                 \nModules to choose from: " << Modules << endl;
        cerr << "Use --help or -h as the first argument to print the help message.\n";
        return -1;
    }
    
    string module_name = string(argv[1]);
    
    if (module_name == "azi_inc")
        cout << "azi_inc\n";
    else if (module_name == "fit_orbit")
        return fit_orbit(argc, argv);
    else if (module_name == "eval_orbit")
        cout << "eval_orbit\n";
    else {
        cerr << "Unrecognized module: "<< module_name << endl;
        cerr << "Modules to choose from: " << Modules << endl;
        return -1;
    }
}
