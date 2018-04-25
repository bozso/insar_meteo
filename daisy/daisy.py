#!/usr/bin/env python3

import argparse
from subprocess import call, STDOUT, CalledProcessError
from shlex import split

_steps = ["data_select", "dominant", "poly_orbit", "integrate", "zero_select"]

_steps__doc__ = ", ".join(_steps)

_daisy__doc__=\
"""
DAISY
Steps: [{}]
""".format(", ".join(_steps))


def cmd(module, *args):
    """
    Calls a daisy module. Arbitrary number of arguments can be passed through
    `*args`. See documentation of modules' for arguments.
    The passed arguments will be converted to string joined and appended to the
    command.
    
    Parameters
    ----------
    module : str
        Name of the daisy module to be called.
    
    *args : list
        Arbitrary number of arguments.
    
    Returns
    -------
    ret_code : int
        Code returned by daisy module.
    
    Raises
    ------
    CalledProcessError
        If something went wrong with the calling of the daisy module.
    ValueError
        If return code if not zero.
    """
    command = "daisy {} {}".format(module, " ".join(str(arg) for arg in args))
    
    try:
        ret_code = call(split(command), stderr=STDOUT)
    except CalledProcessError as e:
        print("Command failed, command: '{}'".format(command))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))

        ret_code = e.returncode
        print("RETURNCODE: \n{}".format(ret_code))
        
        exit(ret_code)
    
    if ret_code != 0:
        raise ValueError("Non zero returncode ({}) from command: {}"
                         .format(ret_code, command))
    
    return ret_code
    
def data_select(in_asc, in_dsc, ps_sep=100.0):
    cmd("data_select", in_asc, in_dsc, ps_sep)

def dominant(in_asc="asc_data.xys", in_dsc="dsc_data.xys", ps_sep=100.0):
    cmd("dominant", in_asc, in_dsc, ps_sep)

def poly_orbit(asc_orbit="asc_master.res", dsc_orbit="dsc_master.res", deg=4):
    cmd("poly_orbit", asc_orbit, deg)
    cmd("poly_orbit", dsc_orbit, deg)

def integrate(dominant="dominant.xyd", asc_fit_orbit="asc_master.porb",
              dsc_fit_orbit="dsc_master.porb"):
    cmd("integrate", dominant, asc_fit_orbit, dsc_fit_orbit)

def zero_select(integrate="integrate.xy", zero_crit=0.6):
    cmd("zero_select", integrate, zero_crit)

def parse_args():

    parser = argparse.ArgumentParser(
            description=_daisy__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("in_asc", help="text file that contains the "
                        "ASCENDING PS velocities")
    parser.add_argument("in_dsc", help="text file that contains the "
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
    
    parser.add_argument("--ps_sep", help="Maximum separation distance "
                        "between ASC and DSC PS points in meters.",
                        nargs="?", type=float,
                        default=100.0)
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
    
    args = parse_args()
    
    start, stop = parse_steps(args)
    ps_sep = args.ps_sep
    
    if start == 0:
        data_select(args.in_asc, args.in_dsc, ps_sep=ps_sep)
    
    return 0
    
if __name__ == "__main__":
    main()
