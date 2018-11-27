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

from argparse import ArgumentParser
from subprocess import check_output, CalledProcessError
from shlex import split
from logging import getLogger
from inmet.inmet_aux import ell_to_merc_full, ell_to_merc_fast 

log = getLogger("inmet.utils")

ellipsoids = {
    "mercator": (6378137.0, 8.1819190903e-2)
}


def ell2merc(lon, lat, isdeg=True, ellipsoid="mercator", lon0=None, fast=False):
    
    if lon0 is None:
        lon0 = np.mean(lon)
    
    ell = ellipsoids[ellipsoid]
    
    return ell_to_merc_fast(lon, lat, lon0, ell[0], ell[1], isdeg, fast), lon0


def _make_cmd(command):
    def f(*args, debug=False):
        Cmd = command + " " + " ".join(proc_arg(arg) for arg in args)
        
        log.debug("Issued command is \"{}\"".format(Cmd))
        
        if debug:
            print(Cmd)
            return
        
        try:
            proc = subprocess.check_output(split(Cmd), stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log.error("\nNon zero returncode from command: \n'{}'\n".format(Cmd))
            log.error("\nOUTPUT OF THE COMMAND: \n\n{}".format(e.output.decode()))
            log.error("\nRETURNCODE was: {}".format(e.returncode))
    
            raise e
    
        return proc
    return f


_inmet_commands = ("fit_orbit", "eval_orbit", "azi_inc")

im = type("Inmet", (object,),
          dict((cmd, _make_cmd(cmd)) for cmd in _inmet_commands))


def cmd(Cmd, *args, debug=False):
    """
    Calls a command line program. Arbitrary number of arguments can be passed
    through `*args`. See documentation of modules for arguments.
    The passed arguments will be converted to string joined together and
    appended to the command.
    
    Parameters
    ----------
    Cmd : str
        Name of the program to be called.
    
    *args
        Arbitrary number of arguments, that can be converted to string, i.e.
        they have a __str__ method.
    
    Returns
    -------
    ret : byte-string
        Output of called module.
    
    Raises
    ------
    CalledProcessError
        If something went wrong with the calling of the module, e.g.
        non zero returncode.

    Examples
    --------
    
    >>> from inmet.cwrap import cmd
    >>> cmd("ls", "*.png", "*.txt")
    """
    
    Cmd = "{} {}".format(Cmd, " ".join(str(arg) for arg in args))
    
    if debug:
        print(Cmd)
        return None
    
    try:
        cmd_out = check_output(split(Cmd), stderr=sub.STDOUT)

    except CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(Cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
        
        raise e
        
    return cmd_out


def gen_sequence(elem_cast, sequence_cast=tuple):
    """
    Returns a function that creates a sequence of elements found in x.
    Converts x values using the passed elem_cast function.
    Helper function for parsing command line arguments.
    """
    return lambda x: sequence_cast(elem_cast(elem) for elem in x.split(","))


def parse_steps(args, steps):
    
    step = args.get("step")
    
    if step is not None:
        if step not in steps:
            raise ValueError("{} is not a valid step.".format(step))
        return [step]
    else:
        start = args.get("start")
        stop  = args.get("stop")

        if start not in steps:
            raise ValueError("{} is not a valid step.".format(start))
        
        if stop not in steps:
            raise ValueError("{} is not a valid step.".format(stop))
        
        first = steps.index(start)
        last = steps.index(stop)

        return steps[first:last + 1]


def gen_step_parser(steps):
    
    parser = ArgumentParser(add_help=False)
    
    parser.add_argument(
        "--step",
        nargs="?",
        default=None,
        type=str,
        choices=steps,
        help="Carry out the processing step defined by this argument "
             "and exit.")

    parser.add_argument(
        "--start",
        nargs="?",
        default=steps[0],
        type=str,
        choices=steps,
        help="Starting processing step. Processing steps will be executed "
             "until processing step defined by --stop is reached.")
    
    parser.add_argument(
        "--stop",
        nargs="?",
        default=steps[-1],
        type=str,
        choices=steps,
        help="Last processing step to be executed.")
    
    return parser

def get_par(parameter, search):

    if isinstance(search, list):
        searchfile = search
    elif pth.isfile(search):
        with open(search, "r") as f:
            searchfile = f.readlines()
    else:
        raise ValueError("search should be either a list or a string that "
                         "describes a path to the parameter file.")
    
    parameter_value = None
    
    for line in searchfile:
        if parameter in line:
            parameter_value = " ".join(line.split(":")[1:]).strip()
            break

    return parameter_value

_default_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_log(logger_name, filename=None, formatter=_default_log_format,
              loglevel="debug"):
    
    logger = logging.getLogger(logger_name)
    
    level = getattr(logging, loglevel.upper(), None)
    
    if not isinstance(level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    
    logger.setLevel(level)

    form = logging.Formatter(formatter, datefmt="%Y.%m.%d %H:%M:%S")

    
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(form)
        logger.addHandler(fh)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(form)
    
    logger.addHandler(consoleHandler)    

    return logger
