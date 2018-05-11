import numpy as np
try:
    from tkinter import *
except ImportError:
    from Tkinter import *

class Plotter(object):
    def __init__(self, root, ax_lim, width=450, height=350, bg="white",
                 xticks=10, yticks=10, xpad=50, ypad=50, ax_width=2,
                 xlabel=None, ylabel=None, form="{:3.2f}",
                 font="Helvetica 12", ticksize=5, xoff=10, yoff=20):
    
        self.width = width
        self.height = height
        self.ax_lim = ax_lim
        
        self.xpad, self.ypad = xpad, ypad
        
        x0, y0 = xpad, height - ypad
        self.x0 = x0
        self.y0 = y0
        
        cv = Canvas(width=width, height=height, bg=bg)
        cv.pack()
        
        cv.create_line(x0, y0, width - xpad, y0, width=ax_width)
        cv.create_line(x0, ypad, x0, y0, width=ax_width)
        
        x_range = float(ax_lim[1] - ax_lim[0])
        y_range = float(ax_lim[3] - ax_lim[2])
        
        xratio = (width  - 2 * xpad) / x_range
        yratio = (height - 2 * ypad) / y_range
        
        self.xratio, self.yratio = xratio, yratio
        
        # create xticks
        if isinstance(xticks, int):
            dx_real = x_range / float(xticks)
            dx_cv   = (width - 2 * xpad) / float(xticks)
            x0_real = ax_lim[0]
            
            for ii in range(xticks):
                x = x0 + ii * dx_cv
                xx = x0_real + ii * dx_real
                cv.create_line(x, y0, x, y0 - ticksize, width=ax_width)
                cv.create_text(x, y0 + xoff, font=font, text=form.format(xx))
        else:
            nticks = len(xticks)

            for ii in range(nticks):
                xx = xticks[ii]
                x = width - xpad - xratio * (ax_lim[1] - xx)
                cv.create_line(x, y0, x, y0 - ticksize, width=ax_width)
                cv.create_text(x, y0 + xoff, font=font, text=form.format(xx))

        # create yticks
        if isinstance(yticks, int):
            dy_real = y_range / float(yticks)
            dy_cv   = (height - 2 * ypad) / float(yticks)
            y0_real = ax_lim[2]
            
            for ii in range(yticks):
                y = y0 - ii * dy_cv
                yy = y0_real + ii * dy_real
                cv.create_line(x0, y, x0 + ticksize, y, width=ax_width)
                cv.create_text(x0 - yoff, y, font=font, text=form.format(yy))
        else:
            nticks = len(yticks)
            ratio = (height - 2 * ypad) / y_range

            for ii in range(nticks):
                yy = yticks[ii]
                y = ypad + yratio * (ax_lim[3] - yy)
                cv.create_line(x0, y, x0 + ticksize, y, width=ax_width)
                cv.create_text(x0 - yoff, y, font=font, text=form.format(yy))
        
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
    
    def plot_data(self, x, y, lines=False, line_fill="black",
                  point_fill="SkyBlue2", point_size=6, point_width=2,
                  line_width=2):
        
        width, height = self.width, self.height
        xratio, yratio = self.xratio, self.yratio
        xpad, ypad = self.xpad, self.ypad
        ax_lim = self.ax_lim
        
        cv = self.cv
        
        scaled = [(width - xpad - xratio * (ax_lim[1] - xx),
                   ypad + yratio * (ax_lim[3] - yy))
                   for (xx, yy) in zip(x, y)]
        
        if lines:
            cv.create_line(scaled, fill=line_fill, width=line_width)
        
        for (x, y) in scaled:
            print(x, y)
            cv.create_oval(x - point_size, y - point_size,
                           x + point_size, y + point_size,
                           outline="black", fill=point_fill,
                           width=point_width)
                            
        
    def save_ps(self, outfile, font="Arial", fontsize=12):
        ps = self.cv.postscript(fontmap="{} {}".format(font, fontsize))
        
        with open(outfile, 'w') as f:
            f.write(ps)
