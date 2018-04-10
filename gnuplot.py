import subprocess as sub
import numpy as np

class Gnuplot(object):
    def __init__(self, out=None, term="xterm", persist=False):
        self.commands = []
        self.is_persist = persist
        self.is_multi = False
        self.is_exec = False
        
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
    
    def list2str(self, command):
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
        for elems in zip(*command)]
        temp.append("e".encode())
        
        return b"\n".join(temp.encode())
    
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
        temp.append("e".encode())
        
        return b"\n".join(temp.encode())

    # SETTERS
    
    def plot(self, *args):
        plot_cmd = ",".join([text for _, text in args])
        
        self.commands.append("plot {}".format(plot_cmd).encode())
        self.commands.extend([array.tobytes() for array, _ in args])
        
        print(self.commands)
        
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
        # print(self.commands)
        if not self.is_exec:
            if self.is_persist:
                g_cmd = ["gnuplot", "--persist"]
            else:
                g_cmd = "gnuplot"
            
            if self.is_multi:
                self.commands.append("unset multiplot".encode())
            
            sub.run(g_cmd, stderr=sub.STDOUT, input=b"\n".join(self.commands),
                    check=True)
        
        del self.is_multi
        del self.is_persist
        del self.commands
        del self.is_exec

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

def plotter(array, pt_type="circ", pt_size=1.0, line_type=None, line_width=1.0,
            linestyle=None, rgb=None, matrix=None, title=None, **kwargs):

    array = np.array(array)
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
        
    text = "'-' "

    if matrix is not None:
        binary = "binary matrix"
    else:
        binary = "binary record={} format='{}'".format(array.shape[0],
                  array.shape[1] * fmt_dict[array.dtype])
    
    text += binary
    
    for key in ["index", "every", "using", "smooth", "axes"]:
        if key in keys:
            text += " {} {}".format(key, kwargs[key])
    
    if linestyle is not None:
        text += " with linestyle {}".format(linestyle)
    else:
        if pt_type is not None:
            text += " with points pt {} ps {}".format(pt_type_dict[pt_type],
                                                      pt_size)
        elif line_type is not None:
            text += "with lines lt {} lw {}".format(line_type_dict[line_type],
                                                    line_width)
    
    if title is not None:
        text += " title '{}'".format(title)
    else:
        text += " notitle"
    
    return array, text
