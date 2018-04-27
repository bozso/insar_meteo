import os
import os.path as pth
import subprocess as sub
import numpy as np
from shlex import split
from math import ceil, sqrt

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
                 config=None, **common_flags):
        
        if pth.isfile("gmt.conf"): os.remove("gmt.conf")
        if pth.isfile("gmt.history"): os.remove("gmt.history")
        
        self.config = _gmt_defaults
        self.is_portrait = portrait
        
        if config is not None:
            self.config.update(config)
        
        with open("gmt.conf", "w") as f:
            f.write(_gmt_defaults_header)
            f.write("\n".join("{} = {}".format(key.upper(), value)
                              for key, value in self.config.items()))
            
        
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
        
        #os.remove("gmt.conf")
        
        del self.commands
        del self.config
        del self.common
        del self.is_gmt5
        del self.debug
        del self.out
    
    def get_config(self, param):
        return self.config[param.upper()]
    
    def multiplot(self, nplots, top_marg, bottom_marg)
        nrows = ceil(sqrt(nplots) - 1)
        nrows = max([1, nrows])
        ncols = ceil(nElems / nrows)
        
        return

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
    else:
        raise ValueError("`data` is not a path to an existing file.")

    # if we have flags parse them
    if len(flags) > 0:
        gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flag))
                               for key, flag in flags.items()])
    
    if is_gmt5:
        Cmd = "gmt info"
    else:
        Cmd = "gmtinfo"
    
    return cmd(Cmd + " " + gmt_flags, ret_out=True).decode()

def get_ranges(data, binary=None, xy_add=None, z_add=None):
    
    if binary is not None:
        ranges = np.genfromtxt(info(data, bi=binary, C=True).split())
    else:
        ranges = np.genfromtxt(info(data, C=True).split())
    
    if xy_add is not None:
        xy_range = (ranges[0] - xy_add, ranges[1] + xy_add,
                    ranges[2] - xy_add, ranges[3] + xy_add)
    else:
        xy_range = ranges[0:4]

    if z_add is not None:
        z_range = (min(ranges[4:]) - z_add, max(ranges[4:]) + z_add)
    else:
        z_range = (min(ranges[4:]), max(ranges[4:]))
        
    return xy_range, z_range

_gmt_defaults_header = r'''
#
# GMT 5.1.2 Defaults file
# vim:sw=8:ts=8:sts=8
#
'''
_gmt_defaults = {
# ********************
# * COLOR Parameters *
# ********************
"COLOR_BACKGROUND": "black",
"COLOR_FOREGROUND": "white",
"COLOR_NAN": 127.5,
"COLOR_MODEL": "none",
"COLOR_HSV_MIN_S": 1,
"COLOR_HSV_MAX_S": 0.1,
"COLOR_HSV_MIN_V": 0.3,
"COLOR_HSV_MAX_V": 1,
# ******************
# * DIR Parameters *
# ******************
"DIR_DATA": "",
"DIR_DCW": "/usr/share/gmt-dcw",
"DIR_GSHHG": "/usr/share/gmt-gshhg",
# *******************
# * FONT Parameters *
# *******************
"FONT_ANNOT_PRIMARY": "14p,Helvetica,black",
"FONT_ANNOT_SECONDARY": "16p,Helvetica,black",
"FONT_LABEL": "14p,Helvetica,black",
"FONT_LOGO": "8p,Helvetica,black",
"FONT_TITLE": "24p,Helvetica,black",
# *********************
# * FORMAT Parameters *
# *********************
"FORMAT_CLOCK_IN": "hh:mm:ss",
"FORMAT_CLOCK_OUT": "hh:mm:ss",
"FORMAT_CLOCK_MAP": "hh:mm:ss",
"FORMAT_DATE_IN": "yyyy-mm-dd",
"FORMAT_DATE_OUT": "yyyy-mm-dd",
"FORMAT_DATE_MAP": "yyyy-mm-dd",
"FORMAT_GEO_OUT": "D",
"FORMAT_GEO_MAP": "ddd:mm:ss",
"FORMAT_FLOAT_OUT": "%.12g",
"FORMAT_FLOAT_MAP": "%.12g",
"FORMAT_TIME_PRIMARY_MAP": "full",
"FORMAT_TIME_SECONDARY_MAP": "full",
"FORMAT_TIME_STAMP": "%Y %b %d %H:%M:%S",
# ********************************
# * GMT Miscellaneous Parameters *
# ********************************
"GMT_COMPATIBILITY": 4,
"GMT_CUSTOM_LIBS": "",
"GMT_EXTRAPOLATE_VAL": "NaN",
"GMT_FFT": "auto",
"GMT_HISTORY": "true",
"GMT_INTERPOLANT": "akima",
"GMT_TRIANGULATE": "Shewchuk",
"GMT_VERBOSE": "compat",
# ******************
# * I/O Parameters *
# ******************
"IO_COL_SEPARATOR": "tab",
"IO_GRIDFILE_FORMAT": "nf",
"IO_GRIDFILE_SHORTHAND": "false",
"IO_HEADER": "false",
"IO_N_HEADER_RECS": 0,
"IO_NAN_RECORDS": "pass",
"IO_NC4_CHUNK_SIZE": "auto",
"IO_NC4_DEFLATION_LEVEL": 3,
"IO_LONLAT_TOGGLE": "false",
"IO_SEGMENT_MARKER": ">",
# ******************
# * MAP Parameters *
# ******************
"MAP_ANNOT_MIN_ANGLE": "20",
"MAP_ANNOT_MIN_SPACING": "0p",
"MAP_ANNOT_OBLIQUE": 1,
"MAP_ANNOT_OFFSET_PRIMARY": "0.075i",
"MAP_ANNOT_OFFSET_SECONDARY": "0.075i",
"MAP_ANNOT_ORTHO": "we",
"MAP_DEFAULT_PEN": "default,black",
"MAP_DEGREE_SYMBOL": "ring",
"MAP_FRAME_AXES": "WESNZ",
"MAP_FRAME_PEN": "thicker,black",
"MAP_FRAME_TYPE": "fancy",
"MAP_FRAME_WIDTH": "5p",
"MAP_GRID_CROSS_SIZE_PRIMARY": "0p",
"MAP_GRID_CROSS_SIZE_SECONDARY": "0p",
"MAP_GRID_PEN_PRIMARY": "default,black",
"MAP_GRID_PEN_SECONDARY": "thinner,black",
"MAP_LABEL_OFFSET": "0.1944i",
"MAP_LINE_STEP": "0.75p",
"MAP_LOGO": "false",
"MAP_LOGO_POS": "BL/-54p/-54p",
"MAP_ORIGIN_X": "1i",
"MAP_ORIGIN_Y": "1i",
"MAP_POLAR_CAP": "85/90",
"MAP_SCALE_HEIGHT": "5p",
"MAP_TICK_LENGTH_PRIMARY": "5p/2.5p",
"MAP_TICK_LENGTH_SECONDARY": "15p/3.75p",
"MAP_TICK_PEN_PRIMARY": "thinner,black",
"MAP_TICK_PEN_SECONDARY": "thinner,black",
"MAP_TITLE_OFFSET": "14p",
"MAP_VECTOR_SHAPE": 0,
# *************************
# * Projection Parameters *
# *************************
"PROJ_AUX_LATITUDE": "authalic",
"PROJ_ELLIPSOID": "WGS-84",
"PROJ_LENGTH_UNIT": "cm",
"PROJ_MEAN_RADIUS": "authalic",
"PROJ_SCALE_FACTOR": "default",
# *************************
# * PostScript Parameters *
# *************************
"PS_CHAR_ENCODING": "ISOLatin1+",
"PS_COLOR_MODEL": "rgb",
"PS_COMMENTS": "false",
"PS_IMAGE_COMPRESS": "deflate,5",
"PS_LINE_CAP": "butt",
"PS_LINE_JOIN": "miter",
"PS_MITER_LIMIT": "35",
"PS_MEDIA": "a4",
"PS_PAGE_COLOR": "white",
"PS_PAGE_ORIENTATION": "portrait",
"PS_SCALE_X": 1,
"PS_SCALE_Y": 1,
"PS_TRANSPARENCY": "Normal",
# ****************************
# * Calendar/Time Parameters *
# ****************************
"TIME_EPOCH": "1970-01-01T00:00:00",
"TIME_IS_INTERVAL": "off",
"TIME_INTERVAL_FRACTION": 0.5,
"TIME_LANGUAGE": "us",
"TIME_UNIT": "s",
"TIME_WEEK_START": "Monday",
"TIME_Y2K_OFFSET_YEAR": 1950
}

_paper_sizes = {
"A0":         [2380, 3368],
"A1":         [1684, 2380],
"A2":         [1190, 1684],
"A3":         [842, 1190],
"A4":         [595, 842],
"A5":         [421, 595],
"A6":         [297, 421],
"A7":         [210, 297],
"A8":         [148, 210],
"A9":         [105, 148],
"A10":        [74, 105],
"B0":         [2836, 4008],
"B1":         [2004, 2836],
"B2":         [1418, 2004],
"B3":         [1002, 1418],
"B4":         [709, 1002],
"B5":         [501, 709],
"archA":      [648, 864],
"archB":      [864, 1296],
"archC":      [1296, 1728],
"archD":      [1728, 2592],
"archE":      [2592, 3456],
"flsa":       [612, 936],
"halfletter": [396, 612],
"note":       [540, 720],
"letter":     [612, 792],
"legal":      [612, 1008],
"11x17":      [792, 1224],
"ledger":     [1224, 792],
}
