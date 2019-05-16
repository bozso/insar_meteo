#!/usr/bin/env python

# Copyright (C) 2018  István Bozsó
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import inmet as im
import numpy as np


def main():
    
    #ap = argp()
    #
    #ap.addargs(
        #Argp.narg("orbit_data", help="", kind="pos"),
        #Argp.narg("preproc", help="", kind="pos"),
        #
        #Argp.narg("deg", help="Degree of fitted polynom.", type=int, default=3),
        #
        #Argp.narg("plot", help="If defined a plot file will be generated."),
        #
        #Argp.narg("nstep", help="Number of steps used to evaluate the polynom.",
                           #type=int, default=100),
        #
        #Argp.narg("centered", help="If set the mean coordinates and time value "
             #"will be subtracted from the coordinates and time values "
             #"before fitting.", kind="flag")
    #)
    
    
    a = np.array([1, 2, 3])
    
    im.test(a)
    return
    
    sat = im.SatOrbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res", "doris")
    
    sat.fit_orbit(deg=2)
    
    print(sat.fit(sat.time))
    
    return 0
    
    #args = ap.parse_args()
    #
    #sat = Satorbit(args.orbit_data, args.preproc)
    #
    #sat.fit_orbit(centered=args.centered, deg=args.deg)
    #sat.save_fit(args.fit_file)
    #
    #plot = args.plot
    #
    #if plot is not None:
        #sat.plot_orbit(plot)
    #
    #return 0
    
if __name__ == "__main__":
    main()
