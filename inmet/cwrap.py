import os
import subprocess as sub
from shlex import split

from gmt import get_version, _gmt_five

def cmd(Cmd, *args):
    """
    Calls a C module. Arbitrary number of arguments can be passed through
    `*args`. See documentation of modules' for arguments.
    The passed arguments will be converted to string joined togerther and
    appended to the command.
    
    Parameters
    ----------
    module : str
        Name of the C module to be called.
    
    *args
        Arbitrary number of arguments, that can be converted to string, i.e.
        they have a __str__ method.
    
    Returns
    -------
    ret_code : int
        Code returned by daisy module.
    
    Raises
    ------
    CalledProcessError
        If something went wrong with the calling of the daisy module, e.g.
        non zero returncode.
    """
    
    command = "{} {}".format(Cmd, " ".join(str(arg) for arg in args))
    
    try:
        ret_code = call(split(command), stderr=STDOUT)
    except CalledProcessError as e:
        print("Command failed, command: '{}'".format(command))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))

        ret_code = e.returncode
        print("RETURNCODE: \n{}".format(ret_code))
        exit(ret_code)
    
    return ret_code

# *****************
# * DAISY Modules *
# *****************

def data_select(in_asc, in_dsc, ps_sep=100.0):
    cmd("daisy data_select", in_asc, in_dsc, ps_sep)

def dominant(in_asc="asc_data.xys", in_dsc="dsc_data.xys", ps_sep=100.0):
    cmd("daisy dominant", in_asc, in_dsc, ps_sep)

def poly_orbit(asc_orbit="asc_master.res", dsc_orbit="dsc_master.res", deg=4):
    cmd("daisy poly_orbit", asc_orbit, deg)
    cmd("daisy poly_orbit", dsc_orbit, deg)

def integrate(dominant="dominant.xyd", asc_fit_orbit="asc_master.porb",
              dsc_fit_orbit="dsc_master.porb"):
    cmd("daisy integrate", dominant, asc_fit_orbit, dsc_fit_orbit)

# ****************
# * Util Modules *
# ****************

def azi_inc(fit_file, coords, mode, outfile, max_iter=1000):
    cmd("inmet_utils azi_inc", fit_file, coords, mode, max_iter, outfile)

def fit_orbit(path, preproc, savefile, deg=3):
    
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
        
        with open("coords.txyz", "w") as f:
            f.write("\n".join(lines[idx + 1:idx + data_num + 1]))
    
    elif preproc == "gamma":
        data_num = [(ii, line) for ii, line in enumerate(lines)
                    if line.startswith("number_of_state_vectors:")]

        if len(data_num) != 1:
            raise ValueError("More than one or none of the lines contains the "
                             "number of datapoints.")
        
        idx = data_num[0][0]
        
        t_first = float(lines[idx + 1].split(":")[1].split()[0])
        t_step  = float(lines[idx + 2].split(":")[1].split()[0])
        
        coords = (get_par("state_vector_position_{}".format(ii + 1), lines)
                  for ii in range(data_num))
        
        print(coords)
    else:
        raise ValueError('preproc should be either "doris" or "gamma" '
                         'not {}'.format(preproc))
    
    ver = get_version()
    
    if ver > _gmt_five:
        cmd = "gmt trend1d "
    
    os.remove("coords.txyz")
    
def get_par(parameter, search, sep=":"):

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
            parameter_value = " ".join(line.split(sep)[1:]).strip()
            break

    return parameter_value
