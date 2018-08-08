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

from __future__ import print_function

import numpy as np
from os.path import isfile
import pickle as pk

from inmet.gnuplot import Gnuplot
from inmet.utils import get_par
import inmet.inmet_aux as ina

class Satorbit(object):
    def __init__(self):
        pass
    
    def read_orbits(self, path, preproc):
    
        if not isfile(path):
            raise IOError("{} is not a file.".format(path))
    
        with open(path, "r") as f:
            lines = f.readlines()
        
        if preproc == "doris":
            data_num = [(ii, line) for ii, line in enumerate(lines)
                        if line.startswith("NUMBER_OF_DATAPOINTS:")]
        
            if len(data_num) != 1:
                raise ValueError("More than one or none of the lines contain the "
                                 "number of datapoints.")
        
            idx = data_num[0][0]
            datanum = int(data_num[0][1].split(":")[1])
            
            data = np.fromstring(''.join(lines[idx + 1:idx + data_num + 1]),
                                 count=data_num * 4, dtype=np.double, sep=" ")\
                                 .reshape((data_num, 4))
    
            time = data[:,0]
            coords = data[:,1:]
        
        elif preproc == "gamma":
            data_num = [(ii, line) for ii, line in enumerate(lines)
                        if line.startswith("number_of_state_vectors:")]
    
            if len(data_num) != 1:
                raise ValueError("More than one or none of the lines contains the "
                                 "number of datapoints.")
            
            idx = data_num[0][0]
            datanum = int(data_num[0][1].split(":")[1])
            
            t_first = float(lines[idx + 1].split(":")[1].split()[0])
            t_step  = float(lines[idx + 2].split(":")[1].split()[0])
            
            time = np.arange(t_first, t_first + 7 * t_step, t_step)
            
            coords = \
            np.asarray([str2orbit(get_par(f"state_vector_position_{ii+1}",
                        lines)) for ii in range(data_num)])
            
        else:
            raise ValueError("preproc should be either \"doris\" or \"gamma\" "
                             "not {}".format(preproc))
        
        self.time = time
        self.coords = coords
        self.datanum = datanum
    
    def fit_orbit(self, centered=True, deg=3):
        
        time, coords = self.time, self.coords
        
        self.centered = centered
        self.deg = deg
        
        self.t_start = np.min(time)
        self.t_stop  = np.max(time)
        
        if centered:
            mean_t = np.mean(time)
            mean_coords = np.mean(coords, axis=0)
    
            time -= mean_t
            to_fit = coords - mean_coords
        else:
            to_fit = coords
            cent = "centered:\t0\n"
        
        design = np.vander(time, deg + 1)
        
        # coeffs[0]: polynom coeffcients are in the columns
        # coeffs[1]: residuals
        # coeffs[2]: rank of design matrix
        # coeffs[3]: singular values of design matrix
        self.coeffs = np.linalg.lstsq(design, to_fit, rcond=None)
        
    def save_fit(self, savefile):
        
        time, coeffs = self.time, self.coeffs
        
        if self.centered:
            cent = "centered:\t1\n"
        else:
            cent = "centered:\t0\n"
            
        with open(savefile, "w") as f:
            f.write(cent)
        
            f.write("Start and stop times.")
            f.write("t_start:\t{}\n".format(self.t_start))
            f.write("t_stop:\t{}\n".format(self.t_stop))
            f.write("Degree of fitted polynom")
            f.write("deg:\t{}\n".format(self.deg))
            
            f.write("(x, y, z) residuals: ({})\n"
                    .format(", ".join(str(elem) for elem in coeffs[1].astype(str))))
    
            f.write("\nCoeffcients are written in a single line from highest to\
                     \nlowest power. First for x than for y and finally for z.\n\n")
            f.write("coefficients:\t{}\n"
                    .format(" ".join(str(elem)
                                for elem in coeffs[0].T.reshape(-1).astype(str))))
            
            if centered:
                f.write("mean_time:\t{}\n".format(mean_t))
                f.write("mean_coords:\t{}\n"
                        .format(" ".join(str(coord) for coord in self.mean_coords)))
        
    def load_fit(self, fit_file):
        
        with open(fit_file, "r") as f:
            poly = {line.split(":")[0]: line.split(':')[1].strip()
                    for line in f if ":" in line}
        
        self.centered = int(poly["centered"])
        self.deg = int(poly["deg"])
        
        if self.centered:
            self.mean_coords = np.genfromtxt(poly["mean_coords"].split(),
                                             dtype=np.double)
            self.t_mean      = float(poly["mean_time"])
        else:
            self.mean_coords = [0.0, 0.0, 0.0]
            self.t_mean      = 0.0
        
        self.t_start = float(poly["t_start"])
        self.t_stop  = float(poly["t_stop"])
        
        self.coeffs = np.genfromtxt(poly["coefficients"].split(),
                                    dtype=np.double).reshape(3, deg + 1)

    def azi_inc(fit_file, coords, is_lonlat=True, max_iter=1000):
        
        return ina.azi_inc(self.coeffs, self.t_start, self.t_stop, self.t_mean,
                           self.mean_coords, self.centered, self.deg,
                           coords, max_iter, is_lonlat)

    #def plot_poly(self, plotfile, nsamp=100):
        
        #time = np.linspace(self.t_start, self.t_stop, nsamp)
        
        #if self.centered:
            #time_cent = time - self.t_mean
            
            #poly = np.asarray([np.polyval(self.coeffs[ii,:], time_cent)
                               #+ mean_coords[ii] for ii in range(3)]).T
            
        #else:
            #poly = np.asarray([np.polyval(self.coeffs[ii,:], time)
                               #for ii in range(3)]).T
        
        #gpt = Gnuplot(out=plotfile, term="pngcairo font ',10'")
        
        #fit = gpt.list2str(time, poly / 1e3)
        #coords = gpt.list2str(self.time, self.coords)
        
        #gpt.axis_format("x", "")
        #gpt("set lmargin 10")
        #gpt("set bmargin 3")
        #gpt("set tics font ',8'")
        
        #gpt.multiplot((3,1), title=f"Fit of orbital vectors - degree of "
                                   #f"polynom: {deg_poly}")
        
        #gpt.ylabel("X [km]")
        #gpt("plot '-' using 1:($2 / 1e3) title 'Orbital coordinates' pt 7, "
                 #"'-' using 1:2 title 'Fitted polynom' with lines")
        #gpt(coords)
        #gpt(fit)
        
        #gpt.ylabel("Y [km]")
        #gpt("plot '-' using 1:($3 / 1e3) notitle pt 7, "
                #" '-' using 1:3 notitle with lines")
        #gpt(coords)
        #gpt(fit)
        
        #gpt("set format x")
        #gpt.labels(x="Time [s]", y="Z [km]")
        #gpt("plot '-' using 1:($4 / 1e3) notitle pt 7, "
                #" '-' using 1:4 notitle with lines")
        #gpt(coords)
        #gpt(fit)
        
        #del gpt

def str2orbit(line):
    line_split = line.split()
    
    return [float(line_split[0]), float(line_split[1]), float(line_split[2])]
