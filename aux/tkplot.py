import numpy as np
from math import ceil, sqrt

_lambda_per_2 = 28.0

try:
    from tkinter import *
except ImportError:
    from Tkinter import *

class Plotter(object):
    def __init__(self, root, ax_lim=None, width=450, height=350, bg="white",
                       nplot=1, grid=0.0, top=0.05, bottom=0.15, left=0.15,
                       right=0.05,**kwargs):
        
        self.width = width
        self.height = height
        self.ax_lim = ax_lim
        self.nplot = 1
        self.grid = grid
        
        self.top, self.bottom, self.left, self.right = \
        round(height * top), round(height * bottom), \
        round(left * width), round(right * width)
        
        self.xratio, self.yratio = None, None
        
        cv = Canvas(width=width, height=height, bg=bg)
        cv.pack()
        self.cv = cv
        
        if ax_lim is not None:
            self.create_axis(ax_lim, **kwargs)
    
    def create_axis(self, ax_lim, xlabel=None, ylabel=None, form="{:3.2f}",
                    font="Ariel 9", xticks=10, yticks=10, close=True,
                    tags="axis", **kwargs):
        nplot = self.nplot
        cv = self.cv
        grid = self.grid
        self.ax_lim = ax_lim
        
        # ax_lim = [round_tick(elem) for elem in ax_lim]
        
        top, bottom ,left, right = self.top, self.bottom, self.left, self.right
        
        nrows = ceil(sqrt(nplot) - 1)
        nrows = max(1, nrows)
        ncols = ceil(nplot / nrows)
        
        # handle additional arguments
        xtick = kwargs.pop("xtick", 10)
        ytick = kwargs.pop("ytick", 10)
        
        ticksize = kwargs.pop("ticksize", 10)

        xoff = kwargs.pop("xoff", 20)
        yoff = kwargs.pop("yoff", 30)
        
        ax_width = kwargs.pop("ax_width", 2)
        
        x_range = float(ax_lim[1] - ax_lim[0])
        y_range = float(ax_lim[3] - ax_lim[2])
        
        width, height = self.width, self.height
        
        # conversion between real and canvas coordinates
        cv_xrange = width  - left - right
        cv_yrange = height - top - bottom
        
        xratio = cv_xrange / float(x_range)
        yratio = cv_yrange / float(y_range)
        
        self.xratio, self.yratio = xratio, yratio

        y_bottom = height - bottom
        x_right  = width - right
        
        # create x and y axis
        cv.create_line(left, y_bottom, x_right, y_bottom, width=ax_width, tags=tags)
        cv.create_line(left, y_bottom, left, top, width=ax_width, tags=tags)

        if close:
            cv.create_line(x_right, y_bottom, x_right, top, width=ax_width,
                           tags=tags)
            cv.create_line(left, top, x_right, top, width=ax_width, tags=tags)
            
        # create xticks
        if isinstance(xtick, int):
            x0_real = ax_lim[0]
            dx_real = float(x_range / xtick) 
            
            # real coordinates
            xx = [x0_real + ii * dx_real for ii in range(xtick)]
        elif isinstance(xtick, float):
            x0_real = ax_lim[0]
            ntick = int(x_range / xtick)
            
            xx = [x0_real + ii * xtick for ii in range(ntick + 1)]
        elif hasattr(xtick, "__iter__"):
            xx = xticks
        else:
            raise ValueError("xtick should be either and integer, float or an "
                             "iterable.")

        x = [width - right - xratio * (ax_lim[1] - X) for X in xx]

        for X, XX in zip(x, xx):
            cv.create_line(X, y_bottom, X, y_bottom - ticksize, width=ax_width,
                           tags=tags)
            cv.create_text(X, y_bottom + xoff, font=font, text=form.format(XX),
                           tags=tags)

        # create yticks
        if isinstance(ytick, int):
            y0_real = ax_lim[2]
            dy_real = float(y_range / ytick) 
            
            # real coordinates
            yy = [y0_real + ii * dy_real for ii in range(ytick)]
        elif isinstance(xtick, float):
            y0_real = ax_lim[2]
            ntick = int(y_range / ytick)
            
            yy = [y0_real + ii * ytick for ii in range(ntick + 1)]
        elif hasattr(ytick, "__iter__"):
            yy = yticks
        else:
            raise ValueError("ytick should be either and integer,float or an "
                             "iterable.")

        y = [top + yratio * (ax_lim[3] - Y) for Y in yy]

        for Y, YY in zip(y, yy):
            cv.create_line(left, Y, left + ticksize, Y, width=ax_width,
                           tags=tags)
            cv.create_text(left - yoff, Y, font=font, text=form.format(YY),
                           tags=tags)
        
        if grid > 0.0:
            for X in x[1:]:
                cv.create_line(X, y_bottom, X, top, width=grid, tags=tags)
            for Y in y[1:]:
                cv.create_line(left, Y, x_right, Y, width=grid, tags=tags)
        
    def xlabel(self, xlabel, off=10, font="Arial 11 bold", tags="xlabel"):
        self.cv.create_text(self.width / 2, self.height - self.bottom / 2 + off,
                            text=xlabel, font=font, tags=tags)
    
    def ylabel(self, ylabel, off=25, font="Arial 11 bold", tags="ylabel"):
        self.cv.create_text(self.left / 2 - off, self.height / 2,
                            text=ylabel, font=font, angle=90, tags=tags)
    
    def reset_axis(self):
        self.cv.delete("axis")
    
    def plot(self, x, y, lines=False, points=True, line_fill="black",
             point_fill="SkyBlue2", point_size=6, point_width=1.25, line_width=2,
             xadd=0.1, yadd=0.1, make_axis=False, tags=None, **kwargs):

        if len(x) != len(y):
            raise ValueError("Input data must have the same number of elements!")

        width, height = self.width, self.height
        left, right, top, bottom = self.left, self.right, self.top, self.bottom
        
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
            xratio = (width  - left - right) / float(ax_lim[1] - ax_lim[0])
            yratio = (height - top - bottom) / float(ax_lim[3] - ax_lim[2])
            
        cv = self.cv

        scaled = [(width - right - xratio * (ax_lim[1] - xx),
                   top + yratio * (ax_lim[3] - yy))
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
        dx = x1 - (self.width - self.right - self.xratio * (self.ax_lim[1] - x2))
        dy = y1 - (self.top + self.yratio * (self.ax_lim[3] - y2))
        
        return sqrt(dx * dx + dy * dy)
        
    def save_ps(self, outfile, font="Arial 12"):
        """ Does not work at the moment for some reason """
        self.cv.postscript(file=outfile, fontmap=font)

def round_tick(step):
    step_frac = step % 1
    step_whole = step - step_frac
    
    diffs = (abs(step_frac - 0.25), abs(step_frac - 0.5),
             abs(step_frac - 1.0), abs(step_frac - 0.0))
    vals = (0.25, 0.5, 1.0, 0.0)
    
    return step_whole + vals[diffs.index(min(diffs))]

class Unwrapper(object):
    def __init__(self, root, year, los, gnss_last, savefile, xadd=0.1, yadd=0.1,
                 xtick=0.125, ytick=10, thresh=10, **kwargs):
        
        self.xtick, self.ytick = xtick, ytick
        self.savefile = savefile
        
        if gnss_last != 0.0:
            self.gnss_last = gnss_last
            self.plot_last = True
        else:
            self.plot_last = False
        
        self.thresh = thresh
        
        plt = Plotter(root, **kwargs)
        
        button_conf = {"borderwidth": 2, "font": "Arial 11 bold"}
        
        b_reset = Button(root, text="Reset", command=self.reset, **button_conf)
        b_reset.pack(side=LEFT)

        b_save = Button(root, text="Save", command=self.save, **button_conf)
        b_save.pack(side=LEFT)
        
        self.last = tuple(los.copy())
        self.los = los
        self.yadd = yadd
        
        self.year0 = round(year[0])
        year = [y - self.year0 for y in year]
        
        min_year, max_year = min(year), max(year)
        X = (max_year - min_year) * xadd
        
        self.yr = [min_year - X, max_year + X]
        
        min_los, max_los = min(*los, gnss_last), max(*los, gnss_last)
        self.min_los, self.max_los = min_los, max_los
        Y = (max_los - min_los) * yadd
        
        ax_lim = [min_year - X, max_year + X, min_los - Y, max_los + Y]
        ax_lim = [round_tick(elem) for elem in ax_lim]
        self.ax_lim0 = ax_lim
        
        plt.create_axis(ax_lim)
        plt.xlabel("Fractional year since {}".format(self.year0))
        plt.ylabel("LOS displacement [mm]")
        
        plt.plot(year, los, point_fill="red", point_size=3, xtick=xtick,
                 ytick=xtick, point_width=1, tags="original")

        if self.plot_last:
            plt.plot([year[0], year[-1]], [0.0, gnss_last], points=False,
                     lines=True, tags="gnss")

        self.year = year
        
        plt.cv.bind("<Button-1>", lambda event: self.add_value(event, _lambda_per_2))
        plt.cv.bind("<Button-3>", lambda event: self.add_value(event, -_lambda_per_2))
        
        self.plt = plt
        
    def add_value(self, event, value):
        x, y = event.x, event.y
        
        dists = tuple((ii, self.plt.calc_dist(x, y, X, Y))
                      for ii, (X,Y) in enumerate(zip(self.year, self.last))
                      if self.plt.calc_dist(x, y, X, Y) < self.thresh)
        
        if not dists:
            return
        else:
            idx, dists = (tuple(elem) for elem in zip(*dists))
            idx = idx[dists.index(min(dists))]
        
        self.last = tuple(elem if ii < idx else elem + value
                          for ii, elem in enumerate(self.last))
        
        self.plt.cv.delete("original", "last", "axis")
        
        if self.plot_last:
            self.plt.cv.delete("gnss")
        
        min_y, max_y = min(min(self.last), self.min_los), \
                       max(max(self.last), self.max_los)
        
        y = (max_y - min_y) * self.yadd
        ax_lim = [self.yr[0], self.yr[1], min_y - y, max_y + y]
        
        ax_lim = [round_tick(elem) for elem in ax_lim]
        
        self.plt.create_axis(ax_lim)

        if self.plot_last:
            self.plt.plot([self.year[0], self.year[-1]], [0.0, self.gnss_last],
                          points=False, lines=True, tags="gnss")
        
        self.plt.plot(self.year, self.last, lines=True, tags="last",
                      xtick=self.xtick, ytick=self.ytick)

        self.plt.plot(self.year, self.los, tags="original", point_fill="red",
                      xtick=self.xtick, ytick=self.ytick, point_width=1, point_size=3)

    def reset(self):
        self.plt.cv.delete("original", "last", "axis")

        if self.plot_last:
            self.plt.cv.delete("gnss")

        self.last = tuple(self.los.copy())
        
        self.plt.create_axis(self.ax_lim0)

        self.plt.plot(self.year, self.los, tags="original", point_fill="red",
                      xtick=self.xtick, ytick=self.ytick, point_width=1,
                      point_size=3)

        if self.plot_last:
            self.plt.plot([self.year[0], self.year[-1]], [0.0, self.gnss_last],
                          points=False, lines=True, tags="gnss")
    
    def save(self):
        with open(self.savefile, "w") as f:
            for yr, los in zip(self.year, self.last):
                f.write("{} {}\n".format(yr + self.yr0, los))
