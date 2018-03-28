import subprocess as sub
import numpy as np

class Gnuplot:
    def __init__(self, out=None, term="wxt", persist=False):
        self.commands = []
        self.persist = persist
        self.is_multi = False
        
        if out:
            self.commands.append("set out '{}'".format(out))
        
        self.commands.append("set term {}".format(term))
        
    def __call__(self, command):
        
        t_cmd = type(command)
        
        if t_cmd == list:
            temp = [" ".join(str(elem) for elem in elems)
                    for elems in zip(*command)]
            temp.append("e")
            self.commands.append("\n".join(temp))

        elif t_cmd == np.ndarray:
            temp = [" ".join(item) for item in command.astype(str)]
            temp.append("e")
            self.commands.append("\n".join(temp))
        else:
            self.commands.append("{}".format(command))
    
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

    def multiplot(self, layout, title="", rowsfirst=True):
        self.is_multi = True
        
        if rowsfirst:
            first = "rowsfirst"
        else:
            first = "colsfirst"
        
        self.commands.append("set multiplot layout {},{} {} title '{}'"
                             .format(layout[0], layout[1], first, title))
        
    def xtics(self, command):
        self.commands.append("set xtics {}".format(command))

    def ytics(self, command):
        self.commands.append("set ytics {}".format(command))

    def ztics(self, command):
        self.commands.append("set ztics {}".format(command))
    
    def style(self, stylenum, styledef):
        self.commands.append("set style line {} {}"
                             .format(stylenum, styledef))
    
    def axis_format(self, axis, _format):
        self.commands.append("set format {} '{}'".format(axis, _format))
    
    def axis_time(self, axis):
        self.commands.append("set {}data time".format(axis))
    
    def timefmt(self, timeformat):
        self.commands.append("set timefmt '{}'".format(timeformat))
    
    def output(self, outfile, term="pngcairo"):
        self.commands.append("set out '{}'".format(outfile))
        self.commands.append("set term {}".format(term))

    def title(self, title):
        self.commands.append("set title '{}'".format(title))

    def term(self, term="wxt"):
        self.commands.append("set term '{}'".format(term))

    def out(self, out_path):
        self.commands.append("set out '{}'".format(out_path))

    def set(self, var):
        self.commands.append("set {}".format(var))

    # LABELS

    def labels(self, x="x", y="y", z=None):
        self.commands.append("set xlabel '{}'".format(x))
        self.commands.append("set ylabel '{}'".format(y))
        
        if z:
            self.commands.append("set zlabel '{}'".format(z))

    def xlabel(self, xlabel="x"):
        self.commands.append("set xlabel '{}'".format(xlabel))

    def ylabel(self, ylabel="y"):
        self.commands.append("set ylabel '{}'".format(ylabel))

    def zlabel(self, zlabel="z"):
        self.commands.append("set zlabel '{}'".format(zlabel))

    # RANGES

    def ranges(self, _xrange=None, _yrange=None, _zrange=None):
        if _xrange and len(_xrange) == 2:
            self.commands.append("set xrange [{}:{}]"
                                  .format(_xrange[0], _xrange[1]))

        if _yrange and len(_yrange) == 2:
            self.commands.append("set yrange [{}:{}]"
                                  .format(_yrange[0], _yrange[1]))

        if _zrange and len(_zrange) == 2:
            self.commands.append("set zrange [{}:{}]"
                                  .format(_zrange[0], _zrange[1]))

    def xrange(self, xmin, xmax):
        self.commands.append("set xrange [{}:{}]".format(xmin, xmax))

    def yrange(self, ymin, ymax):
        self.commands.append("set yrange [{}:{}]".format(ymin, ymax))

    def zrange(self, zmin, zmax):
        self.commands.append("set zrange [{}:{}]".format(zmin, zmax))

    def replot(self):
        self.commands.append("replot")

    def __del__(self):
        if self.persist:
            g_cmd = ["gnuplot", "--persist"]
        else:
            g_cmd = "gnuplot"
        
        #print("\n".join(self.commands))
        if self.is_multi:
            self.commands.append("unset multiplot")
        
        sub.run(g_cmd, stderr=sub.STDOUT, input="\n".join(self.commands),
                encoding="ascii", check=True)
        
        del self.persist
        del self.commands
