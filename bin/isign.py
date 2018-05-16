#!/usr/bin/env python3

import argparse as ap
from tkinter import Tk
from sys import argv

from aux.tkplot import Unwrapper, Plotter, round_tick

_steps = ["unwrap"]

_ISIGN__doc__=\
"""
ISIGN
Steps: [{}]
""".format(", ".join(_steps))

def isign_cmd(module, *args):
    """
    Calls an ISIGN module. Arbitrary number of arguments can be passed through
    `*args`. See documentation of modules' for arguments.
    The passed arguments will be converted to string joined togerther and
    appended to the command.
    
    Parameters
    ----------
    module : str
        Name of the ISIGN module to be called.
    
    *args
        Arbitrary number of arguments, that can be converted to string, i.e.
        they have a __str__ method.
    
    Returns
    -------
    ret_code : int
        Code returned by daisy module.
    
    Raises
    ------
    CalledProcessError
        If something went wrong with the calling of the daisy module, e.g.
        non zero returncode.
    """
    
    command = "isign {} {}".format(module, " ".join(str(arg) for arg in args))
    
    try:
        ret_code = call(split(command), stderr=STDOUT)
    except CalledProcessError as e:
        print("Command failed, command: '{}'".format(command))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))

        ret_code = e.returncode
        print("RETURNCODE: \n{}".format(ret_code))
        exit(ret_code)
    
    return ret_code

def unwrap(infile, savefile, width=750, height=500, grid=0.125):

    with open(infile, "r") as f:
        data = [[float(line.split()[1]), float(line.split()[2])] for line in f]
    
    year, los = [list(elem) for elem in zip(*data)]

    root = Tk()
    root.title(infile)
    
    unw = Unwrapper(root, year, los, argv[2], width=width, height=height,
                    grid=grid)

    root.mainloop()
    
    return 0

def parse_arguments():
    parser = ap.ArgumentParser(description=_isign__doc__,
                formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("in_asc", help="Text file that contains the "
                        "ASCENDING PS velocities.")
    parser.add_argument("in_dsc", help="Text file that contains the "
                        "DESCENDING PS velocities.")

    parser.add_argument("orb_asc", help="text file that contains the "
                        "ASCENDING PS velocities")
    parser.add_argument("orb_dsc", help="text file that contains the "
                        "DESCENDING PS velocities")
    
    parser.add_argument("--step", help="Carry out the processing step defined by "
                       "this argument and exit.", choices=_steps, default=None,
                       nargs="?", type=str)

    parser.add_argument("--start", help="Starting processing step. Processing "
                       "steps will be executed until processing step defined "
                       "by --stop is reached", choices=_steps,
                       default="data_select", nargs="?", type=str)
    parser.add_argument("--stop", help="Last processing step to be executed.",
                       choices=_steps, default="zero_select", nargs="?",
                       type=str)
    
    #parser.add_argument("-p", "--ps_sep", help="Maximum separation distance "
    #                    "between ASC and DSC PS points in meters.",
    #                    nargs="?", type=float, default=100.0)

    parser.add_argument("-d", "--deg", help="Degree of the polynom fitted to "
                        "satellite orbit coordinates.", nargs="?", type=int,
                        default=3)
    """
    parser.add_argument("--logile", help="logfile name ", nargs="?",
                        type=str, default="daisy.log")
    parser.add_argument("--loglevel", help="level of logging ", nargs="?",
                        type=str, default="DEBUG")
    """

    return parser.parse_args()

def parse_steps(args):
    step  = args.step
    start = args.start
    stop  = args.stop
    
    if step is not None:
        first = _steps.index(step)
        last = _steps.index(step)
        return first, last
    else:
        first = _steps.index(start)
        last = _steps.index(stop)
        return first, last

def main():
    
    # args = parse_arguments()
    
    # start, stop = parse_steps(args)
    # ps_sep = args.ps_sep
    
    if len(argv) != 3:
        return
    
    unwrap(argv[1], argv[2])
    
    return
    
    if start == 0:
        data_select(args.in_asc, args.in_dsc, ps_sep=ps_sep)
    
    if start <= 1 and stop >= 1:
        dominant(ps_sep=ps_sep)

    if start <= 2 and stop >= 2:
        poly_orbit(asc_orbit=args.orb_asc, dsc_orbit=args.orb_dsc, deg=args.deg)
    
    if stop == 3:
        integrate()

    return 0
    
if __name__ == "__main__":
    main()
