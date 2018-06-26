#!/usr/bin/env python3

"""

"""

import argparse as ap

import inmet.cwrap as cw

def parse_arguments():
    parser = ap.ArgumentParser(description=__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "orbit_data",
        help="Text file that contains the orbit data.")

    parser.add_argument(
        "preproc",
        help="Type doris or gamma, the program that was used in the "
             "preprocessing of SAR images.")
    
    parser.add_argument(
        "fit_file",
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
    
    cw.fit_orbit(args.orbit_data, args.preproc, args.fit_file, deg=args.deg)
    
    if args.plot is not None:
        cw.plot_orbit(args.orbit_data, args.preproc, args.fit_file, args.plot)
    
    return 0
    
if __name__ == "__main__":
    main()
