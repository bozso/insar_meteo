import os
import subprocess as sub
from shlex import split
import os.path as pth

from inmet.gmt import get_version, _gmt_five, proc_flag, _gmt_commands

def cmd(Cmd, *args, rout=False):
    """
    Calls a C module. Arbitrary number of arguments can be passed through
    `*args`. See documentation of modules for arguments.
    The passed arguments will be converted to string joined together and
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
    ret_out : byte-string
        Output of called module returned if rout=True.
    
    Raises
    ------
    CalledProcessError
        If something went wrong with the calling of the module, e.g.
        non zero returncode.
    """
    
    Cmd = "{} {}".format(Cmd, " ".join(str(arg) for arg in args))
    
    try:
        cmd_out = sub.check_output(split(Cmd), stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(Cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if rout:
        return cmd_out

# *****************
# * DAISY Modules *
# *****************

def data_select(in_asc, in_dsc, ps_sep=100.0, **kwargs):
    cmd("daisy data_select", in_asc, in_dsc, ps_sep, **kwargs)

def dominant(in_asc="asc_data.xys", in_dsc="dsc_data.xys", ps_sep=100.0,
             **kwargs):
    cmd("daisy dominant", in_asc, in_dsc, ps_sep, **kwargs)

def poly_orbit(asc_orbit="asc_master.res", dsc_orbit="dsc_master.res", deg=4,
               **kwargs):
    cmd("daisy poly_orbit", asc_orbit, deg, **kwargs)
    cmd("daisy poly_orbit", dsc_orbit, deg, **kwargs)

def integrate(dominant="dominant.xyd", asc_fit_orbit="asc_master.porb",
              dsc_fit_orbit="dsc_master.porb", **kwargs):
    cmd("daisy integrate", dominant, asc_fit_orbit, dsc_fit_orbit, **kwargs)

# *****************
# * INMET Modules *
# *****************

def azi_inc(fit_file, coords, mode, outfile, max_iter=1000, **kwargs):
    cmd("inmet azi_inc", fit_file, coords, mode, max_iter, outfile, **kwargs)

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
        data_num = int(data_num[0][1].split(":")[1])
        
        with open("coords.txyz", "w") as f:
            f.write("".join(lines[idx + 1:idx + data_num + 1]))
    
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
        
        with open("coords.txyz", "w") as f:
            for ii in range(data_num):
                coords = get_par("state_vector_position_{}".format(ii + 1), lines)
                f.write("{} {}\n".format(t_first + ii * t_step,
                                         coords.split("m")[0]))
    else:
        raise ValueError('preproc should be either "doris" or "gamma" '
                         'not {}'.format(preproc))
    
    ver = get_version()
    
    if ver > _gmt_five:
        Cmd = "gmt trend1d coords.txyz -Np{}r -Fp -V -i0,{}"
    else:
        Cmd = "trend1d coords.txyz -Np{}r -Fp -V -i0,{}"
    
    out = "\n".join(cmd(Cmd.format(deg + 1, ii + 1), rout=True).decode()
                    for ii in range(3))
    
    m = (line.split(":")[2].strip() for ii, line in enumerate(out.split("\n"))
         if "trend1d: Model Coefficients  (Chebyshev):" in line)
    
    with open(savefile, "w") as f:
        f.write("deg: {}\n".format(deg))
        f.write("\n".join(elem.strip()for elem in m) + "\n\n")
        f.write(out)
    
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
