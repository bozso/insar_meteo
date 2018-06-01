#!/usr/bin/env python3

import argparse as ap

import inmet.cwrap as cw

_steps = ["fit_orbit", "azi_inc"]

_daisy__doc__=\
"""
satorbit
Steps: [{}]
""".format(", ".join(_steps))

def parse_arguments():
    parser = ap.ArgumentParser(description=_daisy__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("orbit_data", help="Text file that contains the orbit "
                        "data.")

    parser.add_argument("preproc", help="Type doris or gamma, the program that "
                        "was used in the preprocessing of SAR images.")
    
    parser.add_argument("coords_file", help="Binary file that contains "
                        "either WGS-84 x,y,z or lon., lat., height coordinates "
                        "in double precision (float64).")
    
    parser.add_argument("mode", help="Type xyz for WGS-84 x,y,z or llh for "
                        "WGS-84 lon., lat., height coordinates.")
    
    parser.add_argument("outfile", help="Azimuths and inlcination angles in "
                        "degrees will be saved into this file in binary format.")
    
    parser.add_argument("-d", "--deg", help="Degree of the polynom fitted to "
                        "satellite orbit coordinates.", nargs="?", type=int,
                        default=3)

    parser.add_argument("-m", "--max_iter", help="Maximum iteration number of"
                        "the closest approache method", nargs="?", type=int,
                        default=1000)
    """
    parser.add_argument("--logile", help="logfile name ", nargs="?",
                        type=str, default="daisy.log")
    parser.add_argument("--loglevel", help="level of logging ", nargs="?",
                        type=str, default="DEBUG")
    """

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    cw.fit_orbit(args.orbit_data, args.preproc, "orbit.fit", deg=args.deg)

    cw.azi_inc("orbit.fit", args.coords_file, args.mode, args.outfile,
               max_iter=args.max_iter)

    return 0
    
if __name__ == "__main__":
    main()
