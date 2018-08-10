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

import argparse as ap

import inmet.utils as ut

_steps = frozenset(("data_select", "dominant", "poly_orbit", "integrate"))

_daisy__doc__=\
"""
DAISY
Steps: [{}]
""".format(", ".join(_steps))

def parse_arguments():
    parser = ap.ArgumentParser(description=_daisy__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter,
            parent=[ut.gen_step_parser(_steps)])

    parser.add_argument(
        "in_asc",
        help="Text file that contains the ASCENDING PS velocities.")
        
    parser.add_argument(
        "in_dsc",
        help="Text file that contains the DESCENDING PS velocities.")

    parser.add_argument(
        "orb_asc",
        help="Text file that contains the ASCENDING orbit state vectors.")
    
    parser.add_argument(
        "orb_dsc",
        help="Text file that contains the DESCENDING orbit state vectors.")
    
    parser.add_argument(
        "-p", "--ps_sep",
        nargs="?",
        default=100.0,
        type=float,
        help="Maximum separation distance between ASC and DSC PS points "
             "in meters.")

    parser.add_argument(
        "-d", "--deg",
        nargs="?",
        default=3,
        type=int,
        help="Degree of the polynom fitted to satellite orbit coordinates.")
    
    """
    parser.add_argument("--logile", help="logfile name ", nargs="?",
                        type=str, default="daisy.log")
    parser.add_argument("--loglevel", help="level of logging ", nargs="?",
                        type=str, default="DEBUG")
    """

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    steps = ut.parse_steps(args, _steps)
    ps_sep = args.ps_sep
    
    in_asc = args.in_asc
    in_dsc = args.in_dsc
    
    if "data_select" in steps:
        ut.cmd("daisy data_select", in_asc, in_dsc, ps_sep)

    if "dominant" in steps:
        ut.cmd("daisy dominant", in_asc + "s", in_dsc + "s", ps_sep)

    if "poly_orbit" in steps:
        ut.cmd("daisy poly_orbit", args.orb_asc, args.deg)
        ut.cmd("daisy poly_orbit", args.orb_dsc, args.deg)
        
    if "integrate" in steps:
        ut.cmd("daisy integrate", "dominant.xyd", "asc_master.porb",
               "dsc_master.porb")
    
    return 0
    
if __name__ == "__main__":
    main()
