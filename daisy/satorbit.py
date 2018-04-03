import numpy as np
from gnuplot import Gnuplot

class SatOrbit:
    def __init__(self, filepath, preproc):
        time, coords, data_num = read_orbits(filepath, preproc=preproc)
        
        self.time = time
        self.coords = coords
        self.data_num = data_num
        
        self.mean_t = None
        self.mean_coords = None
        self.is_centered = False
        self.deg = None
        
    def fit_poly(self, deg=3, centered=False):
        
        self.deg = deg
        
        if centered:
            self.mean_t = np.mean(self.time)
            self.mean_coords = np.mean(self.coords, axis=0)
            self.is_centered = True
            
            time = self.time - self.mean_t
            to_fit = self.coords - self.mean_coords
        else:
            to_fit = self.coords
            time = self.time
            self.is_centered = False
        
        self.coeffs = np.asarray([np.polyfit(time, to_fit[:,ii], deg)
                                  for ii in range(3)])
    
    def plot_poly(self, plotfile, nsamp=100):
        time = np.linspace(min(self.time), max(self.time), nsamp)
        
        coeffs = self.coeffs
        deg_poly = self.deg
    
        if self.is_centered:
            time_cent = time - self.mean_t
            mean_coords = self.mean_coords
            
            poly = np.asarray([  np.polyval(coeffs[ii,:], time_cent)
                               + mean_coords[ii] for ii in range(ii)])
            
        else:
            poly = np.asarray([np.polyval(coeffs[ii,:], time)
                               for ii in range(ii)])

        gpt = Gnuplot(out=plotfile, term="pngcairo font ',10'")
    
        fit = gpt.list2str([time, poly / 1e3])
        coords = gpt.list2str([self.time, self.coords])
        
        gpt.axis_format("x", "")
        gpt("set lmargin 10")
        gpt("set bmargin 3")
        gpt("set tics font ',8'")
        
        gpt.multiplot((3,1), title=f"Fit of orbital vectors - degree of "
                                   f"polynom: {deg_poly}")
    
        gpt.ylabel("X [km]")
        gpt("plot '-' using 1:($2 / 1e3) title 'Orbital coordinates' pt 7, "
                 "'-' using 1:2 title 'Fitted polynom' with lines")
        gpt(coords)
        gpt(fit)
    
        gpt.ylabel("Y [km]")
        gpt("plot '-' using 1:($3 / 1e3) notitle pt 7, "
                " '-' using 1:3 notitle with lines")
        gpt(coords)
        gpt(fit)
    
        gpt("set format x")
        gpt.labels(x="Time [s]", y="Z [km]")
        gpt("plot '-' using 1:($4 / 1e3) notitle pt 7, "
                " '-' using 1:4 notitle with lines")
        gpt(coords)
        gpt(fit)
    
        del gpt
    
    def __del__(self):
        del self.time
        del self.coords
        del self.data_num
        del self.mean_t
        del self.mean_coords
        del self.is_centered
        del self.deg


def str2orbit(line):
    line_split = line.split()
    
    return [float(line_split[0]), float(line_split[1]), float(line_split[2])]

def read_orbits(path, preproc="gamma"):

    if not os.path.isfile(path):
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
        data_num = int(data_num[0][1].split(":")[1])
        
        data = np.fromstring(''.join(lines[idx + 1:idx + data_num + 1]),
                             count=data_num * 4, dtype=np.double, sep=" ")\
                             .reshape((data_num, 4)),
        
        return  data[:,0], data[:,1:], data_num
    
    elif preproc == "gamma":
        data_num = [(ii, line) for ii, line in enumerate(lines)
                    if line.startswith("number_of_state_vectors:")]

        if len(data_num) != 1:
            raise ValueError("More than one or none of the lines contains the "
                             "number of datapoints.")
        
        idx = data_num[0][0]
        data_num = int(data_num[0][1].split(":")[1])
        
        t_first = float(lines[idx + 1].split(":")[1].split()[0])
        t_step  = float(lines[idx + 2].split(":")[1].split()[0])
        
        time = np.arange(t_first, t_first + 7 * t_step, t_step)
        
        coords = \
        np.asarray([str2orbit(clu.get_par(f"state_vector_position_{ii+1}",
                    lines)) for ii in range(data_num)])
        
        return time, coords, data_num

    else:
        raise ValueError("preproc should be either 'doris' or 'gamma' "
                         "not {}".format(preproc))
