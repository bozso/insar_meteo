import numpy as np
from math import ceil, sqrt

_lambda_per_2 = 28.0

try:
    from tkinter import *
except ImportError:
    from Tkinter import *

class Plotter(object):
    def __init__(self, root, ax_lim=None, width=450, height=350, bg="white",
                       nplot=1, grid=0.0, xpad=80, ypad=80, **kwargs):
        self.width = width
        self.height = height
        self.ax_lim = ax_lim
        self.nplot = 1
        self.grid = grid
        self.xpad, self.ypad = xpad, ypad
        
        self.x0, self.y0 = None, None
        self.xratio, self.yratio = None, None
        
        cv = Canvas(width=width, height=height, bg=bg)
        cv.pack()
        
        if ax_lim is not None:
            self.create_axis(ax_lim, **kwargs)
        
        self.cv = cv
    
    def create_axis(self, ax_lim, xlabel=None, ylabel=None, form="{:3.2f}",
                    font="Ariel 9", xticks=10, yticks=10, close=True,
                    tags="axis", **kwargs):
        nplot = self.nplot
        cv = self.cv
        grid = self.grid
        self.ax_lim = ax_lim
        
        nrows = ceil(sqrt(nplot) - 1)
        nrows = max(1, nrows)
        ncols = ceil(nplot / nrows)
        
        # handle additional arguments
        xticks = kwargs.pop("xticks", 10)
        yticks = kwargs.pop("yticks", 10)
        
        ticksize = kwargs.pop("ticksize", 10)

        xoff = kwargs.pop("xoff", 20)
        yoff = kwargs.pop("yoff", 30)
        
        ax_width = kwargs.pop("ax_width", 2)
        
        xpad, ypad = self.xpad, self.ypad
        
        x_range = float(ax_lim[1] - ax_lim[0])
        y_range = float(ax_lim[3] - ax_lim[2])
        
        width, height = self.width, self.height
        
        # conversion between real and canvas coordinates
        xratio = (width  - 2 * xpad) / x_range
        yratio = (height - 2 * ypad) / y_range
        
        self.xratio, self.yratio = xratio, yratio

        x0, y0 = xpad, height - ypad
        self.x0 = x0
        self.y0 = y0
        
        # create x and y axis
        cv.create_line(x0, y0, width - xpad, y0, width=ax_width, tags=tags)
        cv.create_line(x0, ypad, x0, y0, width=ax_width, tags=tags)

        if close:
            cv.create_line(width - xpad, y0, width - xpad, ypad,
                           width=ax_width, tags=tags)
            cv.create_line(xpad, ypad, width - ypad, ypad, width=ax_width,
                           tags=tags)
            
        # create xticks
        if isinstance(xticks, int):
            dx_real = x_range / float(xticks)
            dx_cv   = (width - 2 * xpad) / float(xticks)
            x0_real = ax_lim[0]
            
            # real coordinates (x) and canvas coordinates (xx)
            xx = [x0_real + ii * dx_real for ii in range(xticks)]
            x  = [x0 + ii * dx_cv for ii in range(xticks)]
        else:
            x = [width - xpad - xratio * (ax_lim[1] - xx) for xx in xticks]
            xx = xticks

        for X, XX in zip(x, xx):
            cv.create_line(X, y0, X, y0 - ticksize, width=ax_width, tags=tags)
            cv.create_text(X, y0 + xoff, font=font, text=form.format(XX),
                           tags=tags)

        # create yticks
        if isinstance(yticks, int):
            dy_real = y_range / float(yticks)
            dy_cv   = (height - 2 * ypad) / float(yticks)
            y0_real = ax_lim[2]

            y  = [y0 - ii * dy_cv for ii in range(yticks)]
            yy = [y0_real + ii * dy_real for ii in range(yticks)]
        else:
            y = [ypad + yratio * (ax_lim[3] - yy) for yy in yticks]
            yy = yticks

        for Y, YY in zip(y, yy):
            cv.create_line(x0, Y, x0 + ticksize, Y, width=ax_width, tags=tags)
            cv.create_text(x0 - yoff, Y, font=font, text=form.format(YY),
                           tags=tags)
        
        if grid > 0.0:
            for X in x[1:]:
                cv.create_line(X, y0, X, ypad, width=grid, tags=tags)
            for Y in y[1:]:
                cv.create_line(x0, Y, width - xpad, Y, width=grid, tags=tags)
        
    def xlabel(self, xlabel, off=5, font="Arial 11 bold", tags="xlabel"):
        self.cv.create_text(self.width / 2, self.height - self.ypad / 2 + off,
                            text=xlabel, font=font, tags=tags)
    
    def ylabel(self, ylabel, off=25, font="Arial 11 bold", tags="ylabel"):
        self.cv.create_text(self.xpad / 2 - off, self.height / 2,
                            text=ylabel, font=font, angle=90, tags=tags)
    
    def reset_axis(self):
        self.cv.delete("axis")
    
    def plot(self, x, y, lines=False, points=True, line_fill="black",
             point_fill="SkyBlue2", point_size=6, point_width=1.25, line_width=2,
             xadd=0.1, yadd=0.1, make_axis=True, tags=None, **kwargs):

        if len(x) != len(y):
            raise ValueError("Input data must have the same number of elements!")

        width, height = self.width, self.height
        xpad, ypad = self.xpad, self.ypad
        
        if make_axis:
            min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
            
            X = (max_x - min_x) * xadd
            Y = (max_y - min_y) * yadd
            
            ax_lim = [min_x - X, max_x + X, min_y - Y, max_y + Y]
            self.ax_lim = ax_lim

            self.create_axis(ax_lim, **kwargs)
            xratio, yratio = self.xratio, self.yratio
        else:
            ax_lim = self.ax_lim

            # conversion between real and canvas coordinates
            xratio = (width  - 2 * xpad) / float(ax_lim[1] - ax_lim[0])
            yratio = (height - 2 * ypad) / float(ax_lim[3] - ax_lim[2])
            
        cv = self.cv

        scaled = [(width - xpad - xratio * (ax_lim[1] - xx),
                   ypad + yratio * (ax_lim[3] - yy))
                   for xx, yy in zip(x, y)]
        
        # connect dots with lines
        if lines:
            cv.create_line(scaled, fill=line_fill, width=line_width, tags=tags)
        
        if points:
            for x, y in scaled:
                cv.create_oval(x - point_size, y - point_size,
                               x + point_size, y + point_size,
                               outline="black", fill=point_fill,
                               width=point_width, tags=tags)
        
    def calc_dist(self, x1, y1, x2, y2):
        dx = x1 - (self.width - self.xpad - self.xratio * (self.ax_lim[1] - x2))
        dy = y1 - (self.ypad + self.yratio * (self.ax_lim[3] - y2))
        
        return sqrt(dx * dx + dy * dy)
        
    def save_ps(self, outfile, font="Arial 12"):
        """ Does not work at the moment for some reason """
        self.cv.postscript(file=outfile, fontmap=font)

class Unwrapper(object):
    def __init__(self, root, year, los, savefile, xadd=0.1, yadd=0.1, **kwargs):
        
        self.savefile = savefile
        
        plt = Plotter(root, **kwargs)
        
        button_conf = {"borderwidth": 2, "font": "Arial 11 bold"}
        
        b_reset = Button(root, text="Reset", command=self.reset, **button_conf)
        b_reset.pack(side=LEFT)

        b_save = Button(root, text="Save", command=self.save, **button_conf)
        b_save.pack(side=LEFT)
        
        self.last = tuple(los.copy())
        self.los = los
        self.yadd = yadd
        
        year0 = round(year[0])
        year = [y - year0 for y in year]
        self.year0 = year0
        
        min_year, max_year = min(year), max(year)
        X = (max_year - min_year) * xadd
        
        self.yr = [min_year - X, max_year + X]
        self.min_los = min(los)
        self.max_los = max(los)
        
        plt.xlabel("Fractional year since {}".format(year0))
        plt.ylabel("LOS displacement [mm]")
    
        plt.plot(year, los, point_fill="white", point_width=2, tags="original")
        
        self.year = year
        
        plt.cv.bind("<Button-1>", self.add_lambda_per_2)
        plt.cv.bind("<Button-3>", self.subtract_lambda_per_2)
        
        self.plt = plt
        
    def add_lambda_per_2(self, event):
        x, y = event.x, event.y
        
        dists = tuple(self.plt.calc_dist(x, y, X, Y)
                      for X,Y in zip(self.year, self.last))
        idx = dists.index(min(dists))
        
        self.last = tuple(elem if ii < idx else elem + _lambda_per_2
                          for ii, elem in enumerate(self.last))
        
        self.plt.cv.delete("original", "last", "axis")
        
        min_y, max_y = min(min(self.last), self.min_los), \
                       max(max(self.last), self.max_los)
        
        y = (max_y - min_y) * self.yadd
        ax_lim = [self.yr[0], self.yr[1], min_y - y, max_y + y]
        
        self.plt.create_axis(ax_lim)
        self.plt.plot(self.year, self.los, tags="original", point_fill="white",
                      point_width=2, make_axis=False)
        self.plt.plot(self.year, self.last, lines=True, tags="last",
                      make_axis=False)

    def subtract_lambda_per_2(self, event):
        x, y = event.x, event.y
        
        dists = tuple(self.plt.calc_dist(x, y, X, Y)
                      for X,Y in zip(self.year, self.last))
        idx = dists.index(min(dists))
        
        self.last = tuple(elem if ii < idx else elem - _lambda_per_2
                          for ii, elem in enumerate(self.last))
        
        self.plt.cv.delete("original", "last", "axis")
        
        min_y, max_y = min(min(self.last), self.min_los), \
                       max(max(self.last), self.max_los)
        
        y = (max_y - min_y) * self.yadd
        ax_lim = [self.yr[0], self.yr[1], min_y - y, max_y + y]
        
        self.plt.create_axis(ax_lim)
        self.plt.plot(self.year, self.los, tags="original", point_fill="white",
                      point_width=2, make_axis=False)
        self.plt.plot(self.year, self.last, lines=True, tags="last",
                      make_axis=False)

    def reset(self):
        self.plt.cv.delete("original", "last", "axis")
        self.last = tuple(self.los.copy())
        self.plt.plot(self.year, self.los, point_fill="white", point_width=2, tags="original")
    
    def save(self):
        yr0 = self.year0
        
        with open(self.savefile, "w") as f:
            for yr, los in zip(self.year, self.last):
                f.write("{} {}\n".format(yr + yr0, los))
