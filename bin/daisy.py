#!/usr/bin/env python3

import argparse as ap

import inmet.cwrap as cw

_steps = ["data_select", "dominant", "poly_orbit", "integrate"]

_daisy__doc__=\
"""
DAISY
Steps: [{}]
""".format(", ".join(_steps))

def parse_arguments():
    parser = ap.ArgumentParser(description=_daisy__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter,
            parent=[cw.gen_step_parser(_steps)])

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
    
    start, stop = cw.parse_steps(args, _steps)
    ps_sep = args.ps_sep
    
    if start == 0:
        cw.data_select(args.in_asc, args.in_dsc, ps_sep=ps_sep)
    
    if start <= 1 and stop >= 1:
        cw.dominant(ps_sep=ps_sep)

    if start <= 2 and stop >= 2:
        cw.poly_orbit(asc_orbit=args.orb_asc, dsc_orbit=args.orb_dsc, deg=args.deg)
    
    if stop == 3:
        cw.integrate()
    
    return 0
    
if __name__ == "__main__":
    main()
