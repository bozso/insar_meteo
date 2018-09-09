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

from utils import argp, narg
from inmet.satorbit import Satorbit

def main():
    
    ap = argp()
    
    ap.addargs(
        narg("orbit_data", help="", kind="pos"),
        narg("preproc", help="", kind="pos"),
        
        narg("deg", help="Degree of fitted polynom.", type=int, default=3),

        narg("plot", help="If defined a plot file will be generated."),

        narg("nstep", help="Number of steps used to evaluate the polynom.",
                      type=int, default=100),
        
        narg("centered", help="If set the mean coordinates and time value "
             "will be subtracted from the coordinates and time values "
             "before fitting.", kind="flag")
    )
    
    args = ap.parse_args()
    
    sat = Satorbit(args.orbit_data, args.preproc)
    
    sat.fit_orbit(centered=args.centered, deg=args.deg)
    sat.save_fit(args.fit_file)
    
    plot = args.plot
    
    if plot is not None:
        sat.plot_orbit(plot)
    
    return 0
    
if __name__ == "__main__":
    main()
