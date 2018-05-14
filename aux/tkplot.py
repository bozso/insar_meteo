import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *

class Plotter(object):
    def __init__(self, root, ax_lim, width=450, height=350, bg="white",
                 xlabel=None, ylabel=None, form="{:3.2f}", font="Ariel 9",
                 grid=0.0, **kwargs):
        
        # handle additional arguments
        xpad = kwargs.pop("xpad", 80)
        ypad = kwargs.pop("ypad", 50)
        
        xticks = kwargs.pop("xticks", 10)
        yticks = kwargs.pop("yticks", 10)
        
        ticksize = kwargs.pop("ticksize", 5)

        xoff = kwargs.pop("xoff", 10)
        yoff = kwargs.pop("yoff", 20)
        
        ax_width = kwargs.pop("ax_width", 2)
        
        self.width = width
        self.height = height
        self.ax_lim = ax_lim
        
        self.xpad, self.ypad = xpad, ypad
        
        x0, y0 = xpad, height - ypad
        self.x0 = x0
        self.y0 = y0
        
        cv = Canvas(width=width, height=height, bg=bg)
        cv.pack()
        
        # create x and y axis
        cv.create_line(x0, y0, width - xpad, y0, width=ax_width)
        cv.create_line(x0, ypad, x0, y0, width=ax_width)
        
        x_range = float(ax_lim[1] - ax_lim[0])
        y_range = float(ax_lim[3] - ax_lim[2])
        
        # conversion between real and canvas coordinates
        xratio = (width  - 2 * xpad) / x_range
        yratio = (height - 2 * ypad) / y_range
        
        self.xratio, self.yratio = xratio, yratio
        
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
            cv.create_line(X, y0, X, y0 - ticksize, width=ax_width)
            cv.create_text(X, y0 + xoff, font=font, text=form.format(XX))

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
            cv.create_line(x0, Y, x0 + ticksize, Y, width=ax_width)
            cv.create_text(x0 - yoff, Y, font=font, text=form.format(YY))
        
        if grid > 0.0:
            for X in x[1:]:
                cv.create_line(X, y0, X, ypad, width=grid)
            for Y in y[1:]:
                cv.create_line(x0, Y, width - xpad, Y, width=grid)
        

        self.cv = cv
        
    def __del__(self):
        del self.width
        del self.height
        del self.xpad
        del self.ypad
        del self.xratio
        del self.yratio
        del self.ax_lim
        del self.cv
    
    def xlabel(self, xlabel, off=5, font="Arial 10 bold"):
        self.cv.create_text(self.width / 2, self.height - self.ypad / 2 + off,
                            text=xlabel, font=font)
    
    def ylabel(self, ylabel, off=20, font="Arial 10 bold"):
        self.cv.create_text(self.xpad / 2 - off, self.height / 2,
                            text=ylabel, font=font, angle=90)
    
    def plot(self, x, y, lines=False, points=True, line_fill="black",
             point_fill="SkyBlue2", point_size=6, point_width=2, line_width=2):
        
        if len(x) != len(y):
            raise ValueError("Input data must have the same number "
                             "of elements!")
        
        width, height = self.width, self.height
        xratio, yratio = self.xratio, self.yratio
        xpad, ypad = self.xpad, self.ypad
        ax_lim = self.ax_lim
        
        cv = self.cv
        
        # convert real coordinates to canvas coordinates
        scaled = [(width - xpad - xratio * (ax_lim[1] - xx),
                   ypad + yratio * (ax_lim[3] - yy))
                   for xx, yy in zip(x, y)]
        
        # connect dots with lines
        if lines:
            cv.create_line(scaled, fill=line_fill, width=line_width)
        
        if points:
            for x, y in scaled:
                cv.create_oval(x - point_size, y - point_size,
                               x + point_size, y + point_size,
                               outline="black", fill=point_fill,
                               width=point_width)
        
    def save_ps(self, outfile, font="Arial 12"):
        """ Does not work at the moment for some reason """
        self.cv.postscript(file=outfile, fontmap=font)
