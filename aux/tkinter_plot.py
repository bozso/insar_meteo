import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *

class Plot(object):
    def __init__(self, root, ax_lim, width=450, height=350, bg="white",
                 xticks=10, yticks=10, xpad=50, ypad=50, ax_width=2,
                 xlabel=None, ylabel=None, form="{:3.2f}"):
    
        self.width = width
        self.height = height
        self.ax_lim = ax_lim
        
        x0, y0 = xpad, height - ypad
        self.x0 = x0
        self.y0 = y0
        
        cv = Canvas(width=width, height=height, bg=bg)
        cv.pack()
        
        cv.create_line(x0, y0, width - xpad, y0, width=ax_width)
        cv.create_line(x0, ypad, x0, y0, width=ax_width)
        
        x_range = float(ax_lim[1] - ax_lim[0])
        y_range = float(ax_lim[3] - ax_lim[2])
        
        if isinstance(xticks, int):
            dx_real = x_range / float(xticks)
            dx_cv   = (width - 2 * xpad) / float(xticks)
            x0_real = ax_lim[0]
            
            for ii in range(xticks):
                x = x0 + ii * dx_cv
                xx = x0_real + ii * dx_real
                cv.create_line(x, y0, x, y0 - 5, width=ax_width)
                cv.create_text(x, y0 + 5, text=form.format(xx))
        else:
            nticks = len(xticks)
            ratio = (width - 2 * xpad) / x_range

            for ii in range(nticks):
                xx = xticks[ii]
                x = width - xpad - ratio * (ax_lim[1] - xx)
                cv.create_line(x, y0, x, y0 - 5, width=ax_width)
                cv.create_text(x, y0 + 7.5, text=form.format(xx))

        if isinstance(yticks, int):
            dy_real = y_range / float(yticks)
            dy_cv   = (height - 2 * ypad) / float(yticks)
            y0_real = ax_lim[2]
            
            for ii in range(yticks):
                y = y0 - ii * dy_cv
                yy = y0_real + ii * dy_real
                cv.create_line(x0, y, x0 + 5, y, width=ax_width)
                cv.create_text(x0 - 20, y, text=form.format(yy))
        else:
            nticks = len(yticks)
            ratio = (height - 2 * ypad) / y_range

            for ii in range(nticks):
                yy = yticks[ii]
                y = ypad + ratio * (ax_lim[3] - yy)
                cv.create_line(x0, y, x0 + 5, y, width=ax_width)
                cv.create_text(x0 - 20, y, text=form.format(yy))
        
        self.cv = cv
        
    def __del__(self):
        del self.width
        del self.height
        del self.ax_lim
        del self.cv
