# Copyright (C) 2018  MTA CSFK GGI
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

from __future__ import print_function
from builtins import str

import subprocess as sub
import os.path as pth
import numpy as np
import tempfile

from os import remove, popen, fdopen
from math import ceil, sqrt
from sys import stderr

_pt_type_dict = {
    "dot": 0,
    "+": 1,
    "x": 2,
    "+x": 3,
    "empty_square": 4,
    "filed_square": 5,
    "empty_circle": 6,
    "filled_circle": 7,
    "empty_up_triangle": 8,
    "filled_up_triangle": 9,
    "empty_down_triangle": 10,
    "filled_down_triangle": 11,
    "empty_rombus": 12,
    "filled_rombus": 13,
}

_line_type_dict = {
    "black": -1,
    "dashed": 0,
    "red": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
    "teal": 5,
}

class Gnuplot(object):
    def __init__(self, out=None, persist=False, debug=False, silent=False,
                 **kwargs):

        term = str(kwargs.get("term", "wxt"))
        font = str(kwargs.get("font", "Verdena"))
        fontsize = int(kwargs.get("fontsize", 8))
        
        size = kwargs.get("size")
        
        if persist:
            #cmd = ["gnuplot", "--persist"]
            cmd = "gnuplot --persist"
        else:
            #cmd = ["gnuplot"]
            cmd = "gnuplot"
        
        #self.process = sub.Popen(cmd, stderr=sub.STDOUT, stdin=sub.PIPE)
        self.process = popen(cmd, mode="w")
        
        self.write = self.process.write
        self.flush = self.process.flush

        self.persist = persist
        self.debug = debug
        self.silent = silent

        self.multi = False
        self.closed = False

        self.plot_items = []
        self.temps = []
        
        if out:
            self("set out '{}'".format(out))

        term_cmd = "set term {} font '{},{}'".format(term, font, fontsize)

        if size is not None:
            term_cmd += " size {},{}".format(size[0], size[1])

        self(term_cmd)

    def close(self):
        if self.process is not None:
            self.process.close()
            self.process = None

    def __call__(self, command):

        if self.debug:
            stderr.write("gnuplot> {}\n".format(command))
        
        self.write(command + "\n")
        self.flush()        
    
    def refresh(self):
        plot_cmd = self.plot_cmd
        plot_objects = self.plot_items
        
        plot_cmds = ", ".join(plot.command for plot in plot_objects)
        
        if self.debug:
            stderr.write("gnuplot> {} {}\n".format(plot_cmd, plot_cmds))

        self.write(plot_cmd + " " + plot_cmds + "\n")
        
        data = tuple(plot.data for plot in plot_objects if plot.data is not None)
        
        if data:
            self.write("e".join(data) + "\ne\n")
        
        self.flush()

    def _convert_data(self, data, grid=False, **kwargs):
        
        binary = bool(kwargs.get("binary", False))
        temp   = bool(kwargs.get("temp", False))
        
        if binary:
            content = data.tobytes()
            mode = "wb"
            
            if grid:
                add = " binary matrix"
            else:
                add = arr_bin(data)
        else:
            content = np2str(data)
            mode = "w"
            
            if grid:
                add = " matrix"
            else:
                add = ""
        
        if temp:
            if hasattr(tempfile, 'mkstemp'):
                # Use the new secure method of creating temporary files:
                fd, filename, = tempfile.mkstemp(text=True)
                f = fdopen(fd, mode)
            else:
                # for backwards compatibility to pre-2.3:
                filename = tempfile.mktemp()
                f = open(filename, mode)
    
            f.write(content)
            f.close()
            
            self.temps.append(filename)
            
            text  = "'{}'".format(filename)
            array = None
        else:
            text  = "'-'"
            array = content
        
        return array, text + add
        
    def data(self, *arrays, **kwargs):
        
        _check_kwargs(**kwargs)
        
        try:
            data = np.array(arrays).T
        except TypeError:
            raise ValueError("Input arrays should be convertible to a "
                             "numpy array!")
        
        array, text = self._convert_data(data, **kwargs)
        
        text += _parse_line_arguments(**kwargs)
        
        return PlotDescription(array, text)
    
    def grid_data(self, data, x=None, y=None, **kwargs):
        
        _check_kwargs(**kwargs)

        data = np.asarray(data, np.float32)
        
        try:
            (rows, cols) = data.shape
        except ValueError:
            raise DataError("data array must be two-dimensional!")
        
        if x is None:
            x = np.arange(cols)
        else:
            x = np.array(x, dtype=np.float32)
            
            if x.shape != (cols,):
                raise DataError("x should have number of elements equal to "
                                "the number of columns of data!")

        if y is None:
            y = np.arange(rows)
        else:
            y = np.array(y, dtype=np.float32)
            
            if y.shape != (rows,):
                raise DataError("y should have number of elements equal to "
                                "the number of rows of data!")
            
        grid = np.zeros((rows + 1, cols + 1), np.float32)
        grid[0,0] = cols
        grid[0,1:] = x
        grid[1:,0] = y
        grid[1:,1:] = data.astype(np.float32)
        
        array, text = self._convert_data(grid, grid=True, **kwargs)
        
        return PlotDescription(array, text)

    def infile(self, data, matrix=None, binary=None, array=None,
               endian="default", vith=None, **kwargs):
        """
        Sets the text to be used for the 'plot' command of Gnuplot for
        plotting (x,y) data pairs of a file.
        
        Parameters
        ----------
        data : str
            Path to data file.
        pt_type : str, optional
            The symbol used to plot (x,y) data pairs. Default is "circ" (filled
            circles). Selectable values:
                - "circ": filled circles
        pt_size : float, optional
            Size of the plotted symbols. Default value 1.0 .
        line_type : str, optional
            The line type used to plot (x,y) data pairs. Default is "circ"
            (filled circles). Selectable values:
                - "circ": filled circles
        line_width : float, optional
            Width of the plotted lines. Default value 1.0 .
        line_style : int, optional
            Selects a previously defined linestyle for plotting the (x,y) data
            pairs. You can define linestyle with gnuplot.style .
            Default is None.
        title : str, optional
            Title of the plotted datapoints. None (= no title given)
            by default.
        **kwargs
            Additional arguments: index, every, using, smooth, axes.
            See Gnuplot docuemntation for the descritpion of these
            parameters.
        
        Returns
        -------
        cmd: str
        
        Examples
        --------
        
        >>> from gnuplot import Gnuplot
        >>> g = Gnuplot(persist=True)
        >>> g.plot("data1.txt", title="Example plot 1")
        >>> g.plot("data2.txt", title="Example plot 2")
        >>> del g
        
        """
        
        add_keys = ["index", "every", "using", "smooth", "axes"]
        keys = kwargs.keys()    
        
        if not isinstance(data, str) and not pth.isfile(data):
            raise ValueError("data should be a string path to a data file!")
        
        text = "'{}'".format(data)
    
        if binary is not None and not isinstance(binary, bool):
            if array is not None:
                text += " binary array={} format='{}' "\
                        "endian={}".format(array, binary, endian)
            else:
                text += " binary format='{}' endian={}"\
                        .format(binary, endian)
        elif binary:
            text += " binary"
        
        text += " " + " ".join(["{} {}".format(key, kwargs[key])
                               for key in add_keys if key in keys])
        
        text += _parse_line_arguments(**kwargs)
        
        return PlotDescription(None, text)
    
    def plot(self, *plot_objects):
        
        self.plot_items = plot_objects
        self.plot_cmd = "plot"
        
        if not self.silent:
            self.refresh()

    def splot(self, *plot_objects):
        
        self.plot_items = plot_objects
        self.plot_cmd = "splot"
        
        if not self.silent:
            self.refresh()
    
    # ***********
    # * SETTERS *
    # ***********
    
    def size(self, scale, square=False, ratio=None):
        Cmd = "set size"
        
        if sqaure:
            Cmd += " square"
        else:
            Cmd += " no square"
        
        if ratio is not None:
            Cmd += " ratio {}".format(ratio)
        
        Cmd += " {},{}".format(scale[0], scale[1])
        
        self(Cmd)
        
    def palette(self, pal):
        self("set palette {}".format(pal))
    
    def binary(self, definition):
        self("set datafile binary {}".format(definition))

    def margins(self, screen=False, **kwargs):
        
        if screen:
            fmt = "set {} at screen {}"
        else:
            fmt = "set {} {}"
        
        for key, value in kwargs.items():
            if key in ("lmargin", "rmargin", "tmargin", "bmargin"):
                self(fmt.format(key, value))
    
    def multiplot(self, nplot, title=None, nrows=None, order="rowsfirst",
                  portrait=False):

        if nrows is None:
            nrows = ceil(sqrt(nplot) - 1)
            nrows = max([1, nrows])
        
        ncols = ceil(nplot / nrows)
        
        if portrait and ncols > nrows:
            tmp = nrows
            nrows = ncols
            ncols = tmp
            del tmp
        elif nrows > ncols:
            tmp = nrows
            nrows = ncols
            ncols = tmp
            del tmp
        
        self.multi = True
        
        if title is not None:
            self("set multiplot layout {},{} {} title '{}'"\
                 .format(nrows, ncols, order, title))
        else:
            self("set multiplot layout {},{} {}".format(nrows, ncols, order))
    
    def colorbar(self, cbrange=None, cbtics=None, cbformat=None):
        if cbrange is not None:
            self("set cbrange [{}:{}]" .format(cbrange[0], cbrange[1]))
        
        if cbtics is not None:
            self("set cbtics {}".format(cbtics))
        
        if cbformat is not None:
            self("set format cb '{}'".format(cbformat))
        
    def unset_multi(self):
        self("unset multiplot")
        self.multi = False
    
    def reset(self):
        self("reset")
    
    def xtics(self, *args):
        self("set xtics {}".format(_parse_range(*args)))

    def ytics(self, *args):
        self("set ytics {}".format(_parse_range(*args)))

    def ztics(self, *args):
        self("set ztics {}".format(_parse_range(*args)))
    
    def style(self, stylenum, styledef):
        """
        Parameters
        ----------
        """
        self("set style line {} {}".format(stylenum, styledef))
    
    def autoscale(self):
        self("set autoscale")
    
    def axis_format(self, axis, fmt):
        self("set format {} '{}'".format(axis, fmt))
    
    def axis_time(self, axis):
        self("set {}data time".format(axis))
    
    def timefmt(self, fmt):
        self("set timefmt '{}'".format(fmt))
    
    def output(self, outfile, term="pngcairo"):
        self("set out '{}'".format(outfile))
        self("set term {}".format(term))

    def title(self, title):
        self("set title '{}'".format(title))

    def term(self, term="wxt"):
        self("set term '{}'".format(term))

    def out(self, out_path):
        self("set out '{}'".format(out_path))

    def set(self, var):
        self("set {}".format(var))

    def unset(self, var):
        self("unset {}".format(var))

    # LABELS

    def labels(self, x="x", y="y", z=None):
        self("set xlabel '{}'".format(x))
        self("set ylabel '{}'".format(y))
        
        if z is not None:
            self("set zlabel '{}'".format(z))

    def xlabel(self, xlabel="x"):
        self("set xlabel '{}'".format(xlabel))

    def ylabel(self, ylabel="y"):
        self("set ylabel '{}'".format(ylabel))

    def zlabel(self, zlabel="z"):
        self("set zlabel '{}'".format(zlabel))

    # RANGES

    def ranges(self, x=None, y=None, z=None):
        if x is not None and len(x) == 2:
            self("set xrange [{}:{}]".format(x[0], x[1]))

        if y is not None and len(y) == 2:
            self("set yrange [{}:{}]".format(y[0], y[1]))

        if z is not None and len(z) == 2:
            self("set zrange [{}:{}]".format(z[0], z[1]))

    def xrange(self, xmin, xmax):
        self("set xrange [{}:{}]".format(xmin, xmax))

    def yrange(self, ymin, ymax):
        self("set yrange [{}:{}]".format(ymin, ymax))

    def zrange(self, zmin, zmax):
        self("set zrange [{}:{}]".format(zmin, zmax))

    def replot(self):
        self("replot")
    
    def save(self, outfile, term="pngcairo", font="Verdena", fontsize=12,
             enhanced=False):
        
        if enhanced:
            term += " enhanced"
        
        self("set term {} font '{},{}'".format(term, font, fontsize))
        self("set output '{}'".format(outfile))
        
        self.refresh()
        
        self("set term wxt")
        self("set output")
    
    def remove_temps(self):
        for temp in self.temps:
            remove(temp)
        
        self.temps = []
        
    def __del__(self):
        if self.multi:
            self("unset multiplot")
        
        self.close()
        
        for temp in self.temps:
            remove(temp)

class PlotDescription(object):
    def __init__(self, data, command):
        self.data = data
        self.command = command
    
    def __str__(self):
        return "\nData:\n{}\nCommand: {}\n".format(self.data, self.command)

# *************************
# * Convenience functions *
# *************************

def arr_bin(array, image=False):

    fmt_dict = {
        np.dtype("float64"): "%float64",
    }
    
    if array.ndim == 1:
        return " binary format='{}'".format(len(array) * fmt_dict[array.dtype])
    elif array.ndim == 2:
        fmt = array.shape[1] * fmt_dict[array.dtype]
        return " binary record={} format='{}'".format(array.shape[0], fmt)

def np2str(array):
    arr_str = np.array2string(array).replace("[", "").replace("]", "")
    
    return "\n".join(line.strip() for line in arr_str.split("\n"))

def _parse_range(*args):
    if len(args) == 1:
        return str(args[0])
    else:
        return "({})".format(", ".join(str(elem) for elem in args))

def _parse_line_arguments(**kwargs):
    
    linestyle = kwargs.get("linestyle")
    line_type = kwargs.get("line_type")
    line_width = kwargs.get("line_width", 1.0)

    pt_type = kwargs.get("pt_type")
    pt_size = kwargs.get("pt_size", 1.0)
    
    rgb = kwargs.get("rgb")
    title = kwargs.get("title")
    
    text = ""
    
    is_point = pt_type is not None
    is_line  = line_type is not None
    
    if linestyle is not None:
        text += " with linestyle {}".format(linestyle)
    
    elif is_point and is_line:
        text += " with linespoints pt {} ps {} lt {} lw {}"\
                .format(_pt_type_dict[pt_type], pt_size,
                        _line_type_dict[line_type], line_width)
    
    elif is_point:
        text += " with points pt {} ps {}"\
                .format(_pt_type_dict[pt_type], pt_size)
    
    elif is_line:
        text += " with lines lt {} lw {}"\
                .format(_line_type_dict[line_type], line_width)
    
    elif rgb is not None:
        text += " with lines lt {} lw {}".format(rgb, line_width)

    if title is not None:
        text += " title '{}'".format(title)
    else:
        text += " notitle"
    
    return text

def _check_kwargs(**kwargs):

    if not kwargs.get("temp", False) and kwargs.get("binary", False):
        raise OptionError("Inline binary format is not supported!")
    

# **************
# * Exceptions *
# **************

class OptionError(Exception):
    """Raised for unrecognized or wrong option(s)"""
    pass

class DataError(Exception):
    """Raised for data in the wrong format"""
    pass
