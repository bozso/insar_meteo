import subprocess as sub
import numpy as np
import os.path as pth

class Gnuplot(object):
    def __init__(self, out=None, term="xterm", persist=False, debug=False):
        self.commands = []
        self.is_persist = persist
        self.is_multi = False
        self.is_exec = False
        self.debug = False
        
        if out:
            self.commands.append("set out '{}'".format(out).encode())
        
        self.commands.append("set term {}".format(term).encode())
        
    def __call__(self, command):
        
        t_cmd = type(command)
        
        if t_cmd == list:
            temp = [" ".join(str(elem) for elem in elems)
                    for elems in zip(*command)]
            self.commands.extend([temp.encode(), "e".encode()])
            del temp
        elif t_cmd == np.ndarray:
            self.commands.append(command.tobytes())
        else:
            self.commands.append("{}".format(command).encode())
    
    def list2str(self, *command):
        """
        Convert multiple "arrays" stored in separate lists
        to string format, for multiple plotting.
        
        Parameters
        ----------
        command : list
            Contains iterable objects with the same number of elements.
        
        Returns
        -------
        str
            Stringified version of lists.
        """
    
        temp = [" ".join(str(elem) for elem in elems)
                for elems in zip(command)]
        temp.append("e")
        
        return "\n".join(temp)
    
    def np2str(self, command):
        """
        Convert multiple numpy array to string format, for multiple plotting.
        
        Parameters
        ----------
        command : array_like
            Numpy array to be converted to string.
        
        Returns
        -------
        str
            Stringified version numpy array.
        """
        
        temp = [" ".join(item) for item in command.astype(str)]
        temp.append("e")
        
        return "\n".join(temp)

    # SETTERS
    
    def plot(self, *args):
        self.commands.append((",".join(text for text in args)).encode())
        
    def multiplot(self, layout, title="", rowsfirst=True):
        self.is_multi = True
        
        if rowsfirst:
            first = "rowsfirst"
        else:
            first = "colsfirst"
        
        self.commands.append("set multiplot layout {},{} {} title '{}'"
                             .format(layout[0], layout[1], first, title)
                             .encode())

    def unset_multi(self):
        self.commands.append("unset multiplot".encode())
        self.is_multi = False
    
    def reset(self):
        self.commands.append("reset".encode())
    
    def xtics(self, command):
        self.commands.append("set xtics {}".format(command).encode())

    def ytics(self, command):
        self.commands.append("set ytics {}".format(command).encode())

    def ztics(self, command):
        self.commands.append("set ztics {}".format(command).encode())
    
    def style(self, stylenum, styledef):
        """
        Parameters
        ----------
        """
        self.commands.append("set style line {} {}"
                             .format(stylenum, styledef).encode())
    
    def autoscale(self):
        self.commands.append("set autoscale".encode())
    
    def axis_format(self, axis, fmt):
        self.commands.append("set format {} '{}'".format(axis, fmt).encode())
    
    def axis_time(self, axis):
        self.commands.append("set {}data time".format(axis).encode())
    
    def timefmt(self, timeformat):
        self.commands.append("set timefmt '{}'".format(timeformat).encode())
    
    def output(self, outfile, term="pngcairo"):
        self.commands.append("set out '{}'".format(outfile).encode())
        self.commands.append("set term {}".format(term).encode())

    def title(self, title):
        self.commands.append("set title '{}'".format(title).encode())

    def term(self, term="wxt"):
        self.commands.append("set term '{}'".format(term).encode())

    def out(self, out_path):
        self.commands.append("set out '{}'".format(out_path).encode())

    def set(self, var):
        self.commands.append("set {}".format(var).encode())

    # LABELS

    def labels(self, x="x", y="y", z=None):
        self.commands.append("set xlabel '{}'".format(x).encode())
        self.commands.append("set ylabel '{}'".format(y).encode())
        
        if z:
            self.commands.append("set zlabel '{}'".format(z).encode())

    def xlabel(self, xlabel="x"):
        self.commands.append("set xlabel '{}'".format(xlabel).encode())

    def ylabel(self, ylabel="y"):
        self.commands.append("set ylabel '{}'".format(ylabel).encode())

    def zlabel(self, zlabel="z"):
        self.commands.append("set zlabel '{}'".format(zlabel).encode())

    # RANGES

    def ranges(self, x=None, y=None, z=None):
        if x is not None and len(x) == 2:
            self.commands.append("set xrange [{}:{}]"
                                  .format(x[0], x[1]).encode())

        if y is not None and len(y) == 2:
            self.commands.append("set yrange [{}:{}]"
                                  .format(y[0], y[1]).encode())

        if z is not None and len(z) == 2:
            self.commands.append("set zrange [{}:{}]"
                                  .format(z[0], z[1]).encode())

    def xrange(self, xmin, xmax):
        self.commands.append("set xrange [{}:{}]".format(xmin, xmax).encode())

    def yrange(self, ymin, ymax):
        self.commands.append("set yrange [{}:{}]".format(ymin, ymax).encode())

    def zrange(self, zmin, zmax):
        self.commands.append("set zrange [{}:{}]".format(zmin, zmax).encode())

    def replot(self):
        self.commands.append("replot".encode())

    def execute(self):
        if self.is_exec:
            pass
        
        self.is_exec = True
        
        if self.persist:
            g_cmd = ["gnuplot", "--persist"]
        else:
            g_cmd = "gnuplot"
        
        if self.is_multi:
            self.commands.append("unset multiplot".encode())
        
        sub.run(g_cmd, stderr=sub.STDOUT, input=b"\n".join(self.commands),
                check=True)
        
    def __del__(self):
        
        if self.debug
            print((b"\n".join(self.commands)).decode())
        
        if not self.is_exec:
            if self.is_persist:
                g_cmd = ["gnuplot", "--persist"]
            else:
                g_cmd = "gnuplot"
            
            if self.is_multi:
                self.commands.append("unset multiplot".encode())
            
            sub.run(g_cmd, stderr=sub.STDOUT, input=b"\n".join(self.commands),
                    check=True)
        
#-------------------------------------------------------------------
# Convenience functions
#-------------------------------------------------------------------

def arr_bin(array, image=False):
    
    array = np.array(array)
    
    fmt_dict = {
        np.dtype("float64"): "%float64",
    }

    fmt = array.shape[1] * fmt_dict[array.dtype]
    
    return "binary record={} format='{}'".format(array.shape[0], fmt)

def arr_plot(inarray, pt_type="circ", pt_size=1.0, line_type=None,
             line_width=1.0, linestyle=None, rgb=None, matrix=None, title=None,
             **kwargs):
    """
    Sets the text to be used for the 'plot' command of gnuplot for
    plotting (x,y) data pairs of a numpy array.
    
    Parameters
    ----------
    inarray : array_like or list or string
        Input data pairs of the plot. If it is a list, the list elements should
        contain the same number of elements.
    pt_type : str, optional
        The symbol used to plot (x,y) data pairs. Default is "circ" (filled
        circles). Selectable values:
            - "circ": filled circles
    pt_size : float, optional
        Size of the plotted symbols. Default value 1.0 .
    line_type : str, optional
        The line type used to plot (x,y) data pairs. Default is "circ" (filled
        circles). Selectable values:
            - "circ": filled circles
    line_width : float, optional
        Width of the plotted lines. Default value 1.0 .
    line_style : int, optional
        Selects a previously defined linestyle for plotting the (x,y) data
        pairs. You can define linestyle with gnuplot.style . Default is None.
    title : str, optional
        Title of the plotted datapoints. None (= no title given) by default.
    **kwargs
        Additional arguments: index, every, using, smooth, axes. See Gnuplot
        docuemntation for the descritpion of these parameters.
    
    
    Returns
    -------
    
    Examples
    --------
    
    >>> from gnuplot import Gnuplot, arr_plot
    >>> import numpy as np
    >>> g = Gnuplot(is_persist=True)
    >>> array1 = np.array([1, 2], [3, 4]])
    >>> array2 = [[4, 5, 6], [8, 11, 13]]
    >>> g.plot(arr_plot(array1, title="Example plot 1"),
               arr_plot(array2, title="Example plot 2", ))
    
    """
    add_keys = ["index", "every", "using", "smooth", "axes"]
    keys = kwargs.keys()

    fmt_dict = {
        np.dtype("float64"): "%float64",
    }

    pt_type_dict = {
        "circ": 7
    }

    line_type_dict = {
        "circ": 7
    }
        
    arr_type = type(array)
    text = "'-' "
    
    if isinstance(arr_type, list):
        # convert list to string
        temp = [" ".join(str(elem) for elem in elems) for elems in zip(*inarray)]
        temp.append("e".encode())
        
        array = temp
        del temp
    else:
        # try to convert it into a numpy array
        array = np.array(inarray)
    
        if matrix is not None:
            binary = "binary matrix"
        else:
            binary = "binary record={} format='{}'".format(array.shape[0],
                      array.shape[1] * fmt_dict[array.dtype])
    
        text += binary
    
    add_kwargs = " ".join(["{} {}".format(key, kwargs[key])
                           for key in add_keys if key in keys])
    
    text += " {}".format(add_kwargs)
    
    if linestyle is not None:
        text += " with linestyle {}".format(linestyle)
    else:
        if pt_type is not None:
            text += " with points pt {} ps {}".format(pt_type_dict[pt_type],
                                                      pt_size)
        elif line_type is not None:
            text += "with lines lt {} lw {}".format(line_type_dict[line_type],
                                                    line_width)
        elif rgb is not None:
            text += "with lines lt {} lw {}".format(rgb, line_width)
        else:
            raise Exception("Options line_type and rgb are mutually exclusive.")
        
    if title is not None:
        text += " title '{}'".format(title)
    else:
        text += " notitle"
    
    return array, text

def pplot(data, pt_type="circ", pt_size=1.0, line_type=None, line_width=1.0,
         linestyle=None, rgb=None, matrix=None, title=None, binary=None,
         array=None, endian="default", **kwargs):
    """
    Sets the text to be used for the 'plot' command of gnuplot for
    plotting (x,y) data pairs of file.
    
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
        The line type used to plot (x,y) data pairs. Default is "circ" (filled
        circles). Selectable values:
            - "circ": filled circles
    line_width : float, optional
        Width of the plotted lines. Default value 1.0 .
    line_style : int, optional
        Selects a previously defined linestyle for plotting the (x,y) data
        pairs. You can define linestyle with gnuplot.style . Default is None.
    title : str, optional
        Title of the plotted datapoints. None (= no title given) by default.
    **kwargs
        Additional arguments: index, every, using, smooth, axes. See Gnuplot
        docuemntation for the descritpion of these parameters.
    
    Returns
    -------
    cmd: str
    
    Examples
    --------
    
    >>> from gnuplot import Gnuplot, pplot
    >>> g = Gnuplot(is_persist=True)
    >>> g.plot(pplot("data1.txt", title="Example plot 1"),
               pplot("data2.txt", title="Example plot 2"))
    
    """
    add_keys = ["index", "every", "using", "smooth", "axes"]
    keys = kwargs.keys()

    pt_type_dict = {
        "circ": 7
    }

    line_type_dict = {
        "circ": 7
    }
    
    if not isinstance(data, str) or not pth.isfile(data):
        raise ValueError("data should be a string path to a data file")
    
    text = data
    
    if binary is not None:
        if array is not None:
            text += " binary array={} format='{}' "
                    "endian={}".format(array, binary, endian)
        else:
            text += " binary format='{}' endian={}".format(binary, endian)
    
    text += " " + " ".join(["{} {}".format(key, kwargs[key])
                           for key in add_keys if key in keys])
    
    if array is None:
        if linestyle is not None:
            text += " with linestyle {}".format(linestyle)
        else:
            if pt_type is not None:
                text += " with points pt {} ps {}".format(pt_type_dict[pt_type],
                                                          pt_size)
            elif line_type is not None:
                text += "with lines lt {} lw {}".format(line_type_dict[line_type],
                                                        line_width)
            elif rgb is not None:
                text += "with lines lt {} lw {}".format(rgb, line_width)
            else:
                raise Exception("Options line_type and rgb are mutually exclusive.")
        
    if title is not None:
        text += " title '{}'".format(title)
    else:
        text += " notitle"
    
    return text
