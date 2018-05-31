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

    parser.add_argument("outfile", help="Azimuths and inlcination angles in "
                        "degrees will be saved into this file in binary format.")
    
    parser.add_argument("-f", "--fit", help="Text file that contains "
                        "information about the polynoms fitted to orbits.")
    
    #parser.add_argument("--step", help="Carry out the processing step defined by "
                       #"this argument and exit.", choices=_steps, default=None,
                       #nargs="?", type=str)

    #parser.add_argument("--start", help="Starting processing step. Processing "
                       #"steps will be executed until processing step defined "
                       #"by --stop is reached", choices=_steps,
                       #default="data_select", nargs="?", type=str)
    #parser.add_argument("--stop", help="Last processing step to be executed.",
                       #choices=_steps, default="integrate", nargs="?",
                       #type=str)
    
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

def parse_steps(args):
    if step is not None:
        first = _steps.index(args.step)
        last = _steps.index(args.step)
        return first, last
    else:
        first = _steps.index(args.start)
        last = _steps.index(args.stop)
        return first, last
        
def main():
    args = parse_arguments()
    
    #start, stop = parse_steps(args)

    cw.fit_orbit(args.orbit_data, args.preproc, "orbit.fit", deg=args.deg)

    #cw.azi_inc(args.fit_file, args.coords_file, args.outfile,
    #           max_iter=args.max_iter)

    return 0
    
if __name__ == "__main__":
    main()
