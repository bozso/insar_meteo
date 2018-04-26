import os
import os.path as pth
import subprocess as sub
import numpy as np
from shlex import split

# python 2/3 compatibility
from six import string_types

# plotting functions that will use the common output filename, flags and
# the -K and -O flags
_plotters = ["grdcontour", "grdimage", "grdvector", "grdview", "psbasemap",
             "psclip", "pscoast", "pscontour", "pshistogram", "psimage",
             "pslegend", "psmask", "psrose", "psscale", "pstext", "pswiggle",
             "psxy", "psxyz", "gmtlogo"]

class GMT(object):
    def __init__(self, out, gmt5=True, portrait=False, debug=False,
                 **common_flags):
        
        # common output Postscript filename
        self.out = out
        
        # list of sets that contain the gmt commands and its argumants
        # and the filename where the gmt commands output will be written
        self.commands = []
        self.is_gmt5 = gmt5
        self.debug = debug
        
        # if we have common flags parse them
        if len(common_flags) > 0:
            self.common = " ".join(["-{}{}".format(key, proc_flag(flag))
                                    for key, flag in common_flags.items()])
        else:
            self.common = None

    def __del__(self):
        commands = self.commands
        
        idx = [ii for ii, cmd in enumerate(commands) if cmd[0] in _plotters]
        
        # add -K and -O flags to plotter functions
        if len(idx) > 1:
            for ii in idx[:-1]:
                commands[ii] = (commands[ii][0], commands[ii][1] + " -K",
                                commands[ii][2], commands[ii][3])

            for ii in idx[1:]:
                commands[ii] = (commands[ii][0], commands[ii][1] + " -O",
                                commands[ii][2], commands[ii][3])
            
        if self.is_gmt5:
            commands = [("gmt " + cmd[0], *cmd[1:]) for cmd in commands]
        
        if self.debug:
            print("\n".join(" ".join(elem for elem in cmd[0:2])
                  for cmd in commands))
        
        # gather all the outputfiles and remove the ones that already exist
        outfiles = set(cmd[3] for cmd in commands
                       if cmd[3] is not None and pth.isfile(cmd[3]))
        for out in outfiles:
            os.remove(out)
        
        # execute gmt commands and write their output to the specified files
        for cmd in commands:
            execute_gmt_cmd(cmd)
        
        del self.commands
        del self.common
        del self.is_gmt5
        del self.debug
        del self.out
    
    def _gmtcmd(self, gmt_exec, data=None, palette=None, outfile=None, **flags):
        gmt_flags = ""
        
        if data is not None:
            # data is a path to a file
            if isinstance(data, string_types) and pth.isfile(data):
                gmt_flags += "{} ".format(data)
                data = None
            # data is a numpy array
            elif isinstance(data, np.ndarray):
                gmt_flags += "-bi{}dw ".format(data.shape[1])
                data = data.tobytes()
            else:
                raise ValueError("`data` is not a path to an existing file "
                                 "nor is a numpy array.")

        # if we have flags parse them
        if len(flags) > 0:
            gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flag))
                                   for key, flag in flags.items()])
        
        # if we have common flags add them
        if self.common is not None:
            gmt_flags += " " + self.common
        
        if outfile is not None:
            self.commands.append((gmt_exec, gmt_flags, data, outfile))
        else:
            self.commands.append((gmt_exec, gmt_flags, data, self.out))
    
    def __getattr__(self, command, *args, **kwargs):
        def f(*args, **kwargs):
            self._gmtcmd(command, *args, **kwargs)
        return f
    
    def reform(self, portrait=False, **common_flags):
        # if we have common flags parse them
        if len(common_flags) > 0:
            self.common = " ".join(["-{}{}".format(key, proc_flag(flag))
                                    for key, flag in common_flags.items()])
        else:
            self.common = None
    
    def makecpt(self, outfile, **flags):
        keys = flags.keys()
        
        if len(keys) > 0:
            gmt_flags = ["-{}{}".format(key, proc_flag(flags[key]))
                         for key in keys]
        
        self.commands.append(("makecpt", " ".join(gmt_flags), None, outfile))
    
    def Set(self, parameter, value):
        self.commands.append(("set {}={}".format(parameter, value),
                              None, None, None))

# parse GMT flags        
def proc_flag(flag):
    if isinstance(flag, bool) and flag:
        return ""
    elif hasattr(flag, "__iter__") and not isinstance(flag, string_types):
        return "/".join([str(elem) for elem in flag])
    elif isinstance(flag, string_types):
        return flag

def execute_gmt_cmd(cmd, ret_out=False):
    gmt_cmd = cmd[0] + " " + cmd[1]
    
    try:
        cmd_out = sub.check_output(split(gmt_cmd), input=cmd[2],
                                   stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(gmt_cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if cmd[3] is not None:
        with open(cmd[3], "ab") as f:
            f.write(cmd_out)
    
    if ret_out:
        return cmd_out

def cmd(cmd, ret_out=False):
    
    try:
        cmd_out = sub.check_output(split(cmd), stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if ret_out:
        return cmd_out

dtypes = {
    "r4":"f"
}

def get_par(parameter, search):
    search_type = type(search)
    
    if search_type == list:
        searchfile = search
    elif os.path.isfile(search):
        with open(search, "r") as f:
            searchfile = f.readlines()
    else:
        raise ValueError("search should be either a list or a string that "
                         "describes a path to the parameter file.")
    
    parameter_value = None
    
    for line in searchfile:
        if parameter in line and not line.startswith("//"):
            parameter_value = " ".join(line.split()[1:]).split("//")[0].strip()
            break

    return parameter_value

class DEM(object):
    def __init__(self, header_path, endian="small", gmt5=True):
        self.fmt = get_par("SAM_IN_FORMAT", header_path)
        self.dempath = get_par("SAM_IN_DEM", header_path)
        self.nodata = get_par("SAM_IN_NODATA", header_path)
        
        self.is_gmt5 = gmt5
        
        rows, cols = get_par("SAM_IN_SIZE", header_path).split()
        delta_lat, delta_lon = get_par("SAM_IN_DELTA", header_path).split()
        origin_lat, origin_lon = get_par("SAM_IN_UL", header_path).split()
        
        self.nrow, self.ncol, self.delta_lon, self.delta_lat =\
        int(rows), int(cols), float(delta_lon), float(delta_lat)
        
        self.origin_lon, self.origin_lat = float(origin_lon), float(origin_lat)
        
    def __del__(self):
        del self.fmt
        del self.dempath
        del self.nodata
        del self.is_gmt5
        del self.nrow
        del self.ncol
        del self.delta_lon
        del self.delta_lat
        del self.origin_lon
        del self.origin_lat
    
    def make_ncfile(self, ncfile):
        xmin = self.origin_lon
        xmax = self.origin_lon + self.ncol * self.delta_lon

        ymin = self.origin_lat - self.nrow * self.delta_lat
        ymax = self.origin_lat
        
        lonlat_range = "{}/{}/{}/{}".format(xmin ,xmax, ymin, ymax)
        
        increments = "{}/{}".format(self.delta_lon, self.delta_lat)
        
        Cmd = "xyz2grd {infile} -ZTL{dtype} -R{ll_range} -I{inc} -r -G{nc}"\
              .format(infile=self.dempath, dtype=dtypes[self.fmt],
                      ll_range=lonlat_range, inc=increments, nc=ncfile)
        
        if self.is_gmt5:
            Cmd = "gmt " + Cmd
        
        cmd(Cmd)
    
    def plot(self, ncfile, psfile, **gmt_flags):
        
        gmt = GMT(psfile, gmt5=self.is_gmt5)
        gmt.grdimage(data=ncfile, **gmt_flags)
        del gmt

def info(data, is_gmt5=True, **flags):
    gmt_flags = ""
    
    if isinstance(data, string_types) and pth.isfile(data):
        gmt_flags += "{} ".format(data)
        data = None
    # data is a numpy array
    elif isinstance(data, np.ndarray):
        gmt_flags += "-bi{}dw ".format(data.shape[1])
        data = data.tobytes()
    else:
        raise ValueError("`data` is not a path to an existing file "
                         "nor is a numpy array.")

    # if we have flags parse them
    if len(flags) > 0:
        gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flag))
                               for key, flag in flags.items()])
    
    if is_gmt5:
        Cmd = "gmt info"
    else:
        Cmd = "gmtinfo"
    
    return cmd(Cmd + " " + gmt_flags, ret_out=True).decode()
