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

import numpy as np

from argparse import ArgumentParser
from subprocess import check_output, CalledProcessError, STDOUT
from shlex import split
from logging import getLogger
from os.path import join as pjoin
from distutils.ccompiler import new_compiler
from os.path import dirname, realpath, join
from pickle import dump, load
from sys import version_info
from numpy.ctypeslib import _ndptr, ndpointer
from ctypes import *


__all__ = [
    "PY3",
    "CLib",
    "CStruct",
    # "c_arr_p",
    "arrptr",
    "get_filedir",
    "Save",
    "iteraxis",
    "str_t",
    "inarray",
    "outarray",
    "c_idx"
]

           
PY3 = version_info[0] == 3


if PY3:
    str_t = str,
else:
    str_t = basestring,


class Save(object):
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            self = load(f)
        
    
    def save(self, path):
        with open(path, "wb") as f:
            dump(self, f)


def get_filedir():
    return dirname(realpath(__file__))


c_idx = c_long
c_idx_p = POINTER(c_idx)


class CStruct(Structure):
    @classmethod
    def ptr(cls, *args):
        return pointer(cls(*args))
    
    
    def make_ptr(self):
        self.ptr = pointer(self)
    
    
    @classmethod
    def get_ptr_t(cls):
        return POINTER(cls)
    
    
    ptr_t = property(get_ptr_t)
    
    
    @classmethod
    def from_param(cls, obj):
        return pointer(obj)
        
        # try:
        #     return obj.ptr
        # except AttributeError:
        #     obj.make_ptr()
        #     return obj.ptr


class Array(CStruct):
    _fields_ = [
        ("type", c_int),
        ("is_numpy", c_int),
        ("ndim", c_idx),
        ("ndata", c_idx),
        ("datasize", c_idx),
        ("shape", c_idx_p), 
        ("strides", c_idx_p),
        ("data", c_char_p)
    ]
    

    @classmethod
    def from_array(cls, *args, **kwargs):
        tmp = np.array(*args, **kwargs)
        
        ct = tmp.ctypes
        
        return cls(type_conversion[tmp.dtype], 1, c_idx(tmp.ndim),
                   c_idx(tmp.size), c_idx(tmp.itemsize), ct.shape_as(c_idx),
                   ct.strides_as(c_idx), ct.data_as(c_char_p))
    
    
    @classmethod
    def ptr(cls, *args, **kwargs):
        return cls.from_array(*args, **kwargs)

    
    @classmethod
    def from_param(cls, obj):
        return pointer(cls.from_array(obj))


# c_arr_p = POINTER(Array)


def arrptr(**kwargs):
    tmp = ndpointer(**kwargs)
    tmp.from_param = Array.from_param
    
    return tmp


inarray = arrptr(flags=["C_CONTIGUOUS"])
outarray = arrptr(flags=["C_CONTIGUOUS", "OWNDATA", "WRITEABLE"])


lib_filename = new_compiler().library_filename


class CLib(object):
    build_dir = join(get_filedir(), "..", "build")

    def __init__(self, name, path=None):
        if path is None:
            self.path = join(CLib.build_dir, lib_filename(name, lib_type="shared"))
        else:
            self.path = join(path, lib_filename(name, lib_type="shared"))

        self.lib = CDLL(self.path)

    
    def wrap(self, funcname, argtypes, restype=c_int):
        ''' Simplify wrapping ctypes functions '''
        func = getattr(self.lib, funcname)
        func.restype = restype
        func.argtypes = argtypes
        

        def fun(*args):
            ret = func(*args)
            
            if restype is c_int and ret != 0:
                raise RuntimeError('Function "%s" from library "%s" returned '
                                   'with non-zero value!' % (funcname, self.lib))
            
            return ret
            
        return fun


type_conversion = {
    np.dtype(np.int_)        : 1, # C long
    np.dtype(np.intc)        : 2, # C int
    np.dtype(np.intp)        : 3, # C ssize_t

    np.dtype(np.int8)        : 4,
    np.dtype(np.int16)       : 5,
    np.dtype(np.int32)       : 6,
    np.dtype(np.int64)       : 7,

    np.dtype(np.uint8)        : 8,
    np.dtype(np.uint16)       : 9,
    np.dtype(np.uint32)       : 10,
    np.dtype(np.uint64)       : 11,

    np.dtype(np.float32)     : 12,
    np.dtype(np.float64)     : 13,

    np.dtype(np.complex64)   : 14,
    np.dtype(np.complex128)  : 15
}


# log = getLogger("inmet.utils")


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
        cmd_out = check_output(split(Cmd), stderr=STDOUT)

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

    
__default_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_log(logger_name, filename=None, formatter=__default_log_format,
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

    

def iteraxis(a, axis=0):
    """
    A iterator to return slices of a multi-dimensional array over
    the specified axis.
    Parameters
    ----------
    a : array_like
        Input data
    axis : int, optional
        The axis to iterate over.  By default, it is the first axis,
        and thus identical to iterating over the original array.
    Yields
    -------
    out : n ndarrays, of diminsion d-1, where n = a.shape[axis]
          and d = a.ndim
        
    """
    msg = 'iteraxis: %s (%d) must be >=0 and < %d'
    if not (0 <= axis < a.ndim):
        raise ValueError(msg % ('axis', axis, a.ndim))

    a = np.asarray(a)
    return iter(np.rollaxis(a, axis, 0))
