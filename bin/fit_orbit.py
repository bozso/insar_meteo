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

"""
Usage:
    fit_orbit.py <orbit_data> <preproc> <fit_file> [--deg=<d>] [--plot=<file>] 
                 [--nstep=<n>]

Options:
    -h --help      Show this screen.
    --deg=<d>      Degree of fitted polynom [default: 3].
    --plot=<file>  If defined a plot file will be generated.
    --nstep=<n>    Number of steps used to evaluate the polynom [default: 100].
"""

from inmet.satorbit import Satorbit
from inmet.docopt import docopt

def main():
    args = docopt(__doc__)
    
    print(args)
    return 0
    
    sat = Satorbit(args.orbit_data, args.preproc)
    
    sat.fit_orbit(centered=False, deg=args.deg)
    sat.save_fit(args.fit_file)
    
    if args.plot is not None:
        sat.plot_orbit(args.plot)
    
    return 0
    
if __name__ == "__main__":
    main()
