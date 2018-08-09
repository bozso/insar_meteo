#!/usr/bin/env python

# INMET
# Copyright (C) 2018  MTA CSFK GGI
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

"""

import argparse as ap

from inmet.satorbit import Satorbit

def parse_arguments():
    parser = ap.ArgumentParser(description=__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "orbit_data",
        type=str,
        help="Text file that contains the orbit data.")

    parser.add_argument(
        "preproc",
        type=str,
        help="Type doris or gamma, the program that was used in the "
             "preprocessing of SAR images.")
    
    parser.add_argument(
        "fit_file",
        type=str,
        help="Parameters of the fitted polynom will be printed to this ascii file.")
    
    parser.add_argument(
        "-d", "--deg",
        nargs="?",
        default=3,
        type=int,
        help="Degree of the polynom fitted to satellite orbit coordinates.")

    parser.add_argument(
        "--plot",
        nargs="?",
        default=None,
        type=str,
        help="Gnuplot will plot the original and fitted coordinates to this file.")

    parser.add_argument(
        "--nstep",
        nargs="?",
        default=100,
        type=int,
        help="Evaluate fitted x, y, z coordinates at nstep number of steps "
             "between the range of t_min and t_max.")

    """
    parser.add_argument("--logile", help="logfile name ", nargs="?",
                        type=str, default="daisy.log")
    parser.add_argument("--loglevel", help="level of logging ", nargs="?",
                        type=str, default="DEBUG")
    """

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    sat = Satorbit(args.orbit_data, args.preproc)
    
    sat.fit_orbit(centered=False, deg=args.deg)
    sat.save_fit(args.fit_file)
    
    if args.plot is not None:
        sat.plot_orbit(args.plot)
    
    return 0
    
if __name__ == "__main__":
    main()
