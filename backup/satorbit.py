import numpy as np
import os.path as pth

import aux.insar_aux as ina
from aux.gmt import GMT

def fit_orbit(orbit_file, preproc, savefile, centered=True, deg=3):
    
    time, coords, data_num = read_orbits(orbit_file, preproc=preproc)
    
    t_start = time[0]
    t_stop  = time[-1]
    
    if centered:
        mean_t = np.mean(time)
        mean_coords = np.mean(coords, axis=0)
        cent = "centered:\t1\n"

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
    try:
        coeffs = np.linalg.lstsq(design, to_fit, rcond=None)
    except LinAlgError:
        print("Something went wrong in the fitting of the orbit polynom.")
    
    with open(savefile, "w") as f:
        f.write(cent)
        
        f.write("t_start:\t{}\n".format(t_start))
        f.write("t_stop:\t{}\n".format(t_stop))
        f.write("deg:\t{}\n".format(deg))
        
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
                    .format(" ".join(str(coord) for coord in mean_coords)))
        

def azi_inc(fit_file, coords, is_lonlat=True, max_iter=1000):
    
    is_centered, mean_coords, t_mean, t_start, t_stop, coeffs, deg = \
                                                            load_fit(fit_file)
    
    return ina.azi_inc(coeffs, t_start, t_stop, t_mean, mean_coords,
                       is_centered, deg, coords, max_iter, is_lonlat)

def plot_poly(fit_file, orbit_file, processor, plotfile, nsamp=100):
    
    is_centered, mean_coords, t_mean, t_start, t_stop, coeffs, deg = \
                                                            load_fit(fit_file)
    
    t, coords, _ = read_orbits(orbit_file, preproc=processor)
    
    coords = np.hstack([t[:,None], coords])
    
    time = np.linspace(t_start, t_stop, nsamp)
    
    if is_centered:
        time_cent = time - t_mean
        
        poly = np.asarray([np.polyval(coeffs[ii,:], time_cent)
                           + mean_coords[ii] for ii in range(3)]).T
        
    else:
        poly = np.asarray([np.polyval(coeffs[ii,:], time)
                           for ii in range(3)]).T
    
    x_range, y_range, z_range = \
                            (min(poly[:,0]), max(poly[:,0])),\
                            (min(poly[:,1]), max(poly[:,1])),\
                            (min(poly[:,2]), max(poly[:,2]))
    
    gmt = GMT(plotfile, R=(t_start, t_stop, x_range[0], x_range[1]),
                        J="x2i", portrait=True, debug=True)
    
    xx, yy = gmt.multiplot(3, nrows=3)
    
    gmt.psbasemap(Bx="a10g10", By="a1e4g1e4")
    gmt.psxy(data=coords[:,[0,1]], byte_swap=True)
    
    del gmt

def str2orbit(line):
    line_split = line.split()
    
    return [float(line_split[0]), float(line_split[1]), float(line_split[2])]

def read_orbits(path, preproc="gamma"):

    if not pth.isfile(path):
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
        
        data = np.fromstring("".join(lines[idx + 1:idx + data_num + 1]),
                             count=data_num * 4, dtype=np.double, sep=" ")\
                             .reshape((data_num, 4))

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
        np.asarray([str2orbit(get_par(f"state_vector_position_{ii+1}",
                    lines)) for ii in range(data_num)])
        
        return time, coords, data_num

    else:
        raise ValueError("preproc should be either 'doris' or 'gamma' "
                         "not {}".format(preproc))

def get_par(parameter, search):

    if isinstance(search, list):
        searchfile = search
    elif pth.isfile(search):
        with open(search, "r") as f:
            searchfile = f.readlines()
    else:
        raise ValueError("search should be either a list or a string that "
                         "describes a path to the parameter file.")
    
    parameter_value = None
    
    for line in searchfile:
        if parameter in line:
            parameter_value = " ".join(line.split(":")[1:]).strip()
            break

    return parameter_value

def load_fit(fit_file):
    with open(fit_file, "r") as f:
        poly = {line.split(":")[0]: line.split(':')[1].strip()
                for line in f if ":" in line}
    
    is_centered = int(poly["centered"])
    deg = int(poly["deg"])
    
    if is_centered:
        mean_coords = np.genfromtxt(poly["mean_coords"].split(), dtype=np.double)
        t_mean      = float(poly["mean_time"])
    else:
        mean_coords = [0.0, 0.0, 0.0]
        t_mean      = 0.0
    
    t_start = float(poly["t_start"])
    t_stop  = float(poly["t_stop"])
    
    coeffs = np.genfromtxt(poly["coefficients"].split(), dtype=np.double)\
                           .reshape(3, deg + 1)
    
    return is_centered, mean_coords, t_mean, t_start, t_stop, coeffs, deg
    
    
