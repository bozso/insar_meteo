#!/usr/bin/env python3

import argparse as ap
from tkinter import Tk
from sys import argv
from os.path import  isfile

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

def unwrap(ib_pair, line, width=750, height=500, grid=0.125):
    
    sline = line.split()
    gnss_last = float(sline[2])
    infile = ib_pair + "_g_{}.los".format(sline[1].strip())
    
    with open(infile, "r") as f:
        data = [[float(line.split()[1]), float(line.split()[2])] for line in f]
    
    year, los = [list(elem) for elem in zip(*data)]
    
    root = Tk()
    root.title(infile)
    
    if gnss_last == 0.0:
        Unwrapper(root, year, los, infile + ".unw", width=width, height=height,
                  grid=grid)
    else:
        Unwrapper(root, year, los, infile + ".unw",
                  gnss_year=(year[0], year[-1]), gnss_los=(0.0, gnss_last),
                  width=width, height=height,
                  grid=grid)
        
    root.mainloop()
    
    return 0

def parse_arguments():
    parser = ap.ArgumentParser(description=_ISIGN__doc__,
                formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("ib_pair", help="IB pairs to process. E.g. IB3-IB1.",
                        type=str)
    parser.add_argument("mode", help="type asc or dsc to unwrap ascending or "
                        "descending phase values", type=str,
                        choices=("asc", "dsc"))
    parser.add_argument("--width", help="Width of the window.",
                        nargs="?", type=float, default=750)
    parser.add_argument("--height", help="Height of the window.",
                        nargs="?", type=float, default=500)

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
    
    ib_pair = args.ib_pair
    
    ib_file = ib_pair + ".los"
    
    if not isfile(ib_file):
        lines = ["gnss asc 0.0", "gnss dsc 0.0"]
    else:
        with open(ib_file, "r") as f:
            lines = f.readlines()
            
            # swap
            if "dsc" in lines[0]:
                tmp = lines[0]
                lines[0] = lines[1]
                lines[1] = tmp
                
                del tmp
    
    if args.mode == "asc":
        unwrap(ib_pair, lines[0], width=args.width, height=args.height)
    elif args.mode == "dsc":
        unwrap(ib_pair, lines[1], width=args.width, height=args.height)

    return 0
    
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
