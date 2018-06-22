import os
import os.path as pth
import subprocess as subp
from shlex import split
from math import ceil, sqrt
from distutils.version import StrictVersion
from re import sub
from argparse import ArgumentParser

import glob

# plotting functions that will use the common output filename, flags and
# the -K and -O flags
_plotters = ("grdcontour", "grdimage", "grdvector", "grdview", "psbasemap",
             "psclip", "pscoast", "pscontour", "pshistogram", "psimage",
             "pslegend", "psmask", "psrose", "psscale", "pstext", "pswiggle",
             "psxy", "psxyz", "gmtlogo")

_gmt_commands = ('grdcontour', 'grdimage', 'grdvector', 'grdview', 
'psbasemap', 'psclip', 'pscoast', 'pscontour', 'pshistogram', 'psimage', 
'pslegend', 'psmask', 'psrose', 'psscale', 'pstext', 'pswiggle', 'psxy', 
'psxyz', 'gmtlogo', 'blockmean', 'blockmedian', 'blockmode', 'filter1d', 
'fitcircle', 'gmt2kml', 'gmt5syntax', 'gmtconnect', 'gmtconvert', 
'gmtdefaults', 'gmtget', 'gmtinfo', 'gmtlogo', 'gmtmath', 'gmtregress', 
'gmtselect', 'gmtset', 'gmtsimplify', 'gmtspatial', 'gmtvector', 'gmtwhich',
'grd2cpt', 'grd2rgb', 'grd2xyz', 'grdblend', 'grdclip', 'grdcontour', 
'grdconvert', 'grdcut', 'grdedit', 'grdfft', 'grdfilter', 'grdgradient', 
'grdhisteq', 'grdimage', 'grdinfo', 'grdlandmask', 'grdmask', 'grdmath', 
'grdpaste', 'grdproject', 'grdraster', 'grdsample', 'grdtrack', 'grdtrend', 
'grdvector', 'grdview', 'grdvolume', 'greenspline', 'kml2gmt', 'mapproject',
'nearneighbor', 'project', 'sample1d', 'spectrum1d', 'sph2grd', 
'sphdistance', 'sphinterpolate', 'sphtriangulate', 'splitxyz', 'surface', 
'trend1d', 'trend2d', 'triangulate', 'xyz2grd')

_ps2raster = {
".bmp" : "b",
".eps" : "e",
".pdf" : "f",
".jpeg": "j",
".png" : "g",
".ppm" : "m",
".tiff": "t",
}

def gmt(module, *args, ret=False, **kwargs):
    
    if module not in _gmt_commands and "gmt" + module not in _gmt_commands:
        raise ValueError("Unrecognized gmt module: {}".format(module))
    
    if not module.startswith("gmt") and get_version() > _gmt_five:
        module = "gmt " + module
    
    Cmd = module + " " + " ".join(str(arg) for arg in args)

    if len(kwargs) > 0:
        Cmd += " " + " ".join(["-{}{}".format(key, proc_flag(flag))
                               for key, flag in kwargs.items()])
    
    try:
        cmd_out = subp.check_output(split(Cmd), stderr=subp.STDOUT)
    except subp.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(Cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if ret:
        # filter out gmt messages
        return " ".join(elem for elem in cmd_out.decode().split("\n")
                        if not elem.startswith("gmt:"))
    
def gen_tuple(cast):
    """
    Returns a function that creates a tuple of elements found in x.
    Converts x values using the passed cast function.
    Helper function for parsing command line arguments.
    """
    return lambda x: tuple(cast(elem) for elem in x.split(","))

# ***********************************************************
# * Parent argument parsers for creating command line tools *
# ***********************************************************

# Arguemnts for the raster function of the GMT object.

raster_parser = ArgumentParser(add_help=False)

raster_parser.add_argument(
    "--dpi",
    nargs="?",
    default=100,
    type=float,
    help="DPI of the output raster file.")

raster_parser.add_argument(
    "--gray",
    action="store_true",
    help="If defined output raster will be grayscaled.")

raster_parser.add_argument(
    "--portrait",
    action="store_true",
    help="If defined output raster will be in portrait format.")

raster_parser.add_argument(
    "--pagesize",
    action="store_true",
    help="If defined output eps will be created with the PageSize command.")

raster_parser.add_argument(
    "--multi_page",
    action="store_true",
    help="If defined output pdf will be multi paged.")

raster_parser.add_argument(
    "--transparent",
    action="store_true",
    help="If defined output png will be transparent.")

# Arguemnts for the raster multiplot of the GMT object.

multi_parser = ArgumentParser(add_help=False)
    
multi_parser.add_argument(
    "--nrows",
    nargs="?",
    default=None,
    type=int,
    help="Number of rows in multiplot.")

multi_parser.add_argument(
    "--top",
    nargs="?",
    default=0,
    type=float,
    help="Top margin in point units.")

multi_parser.add_argument(
    "--left",
    nargs="?",
    default=50,
    type=float,
    help="Left margin in point units.")

multi_parser.add_argument(
    "--right",
    nargs="?",
    default=125,
    type=float,
    help="Right margin in point units.")

multi_parser.add_argument(
    "--hpad",
    nargs="?",
    default=55,
    type=float,
    help="Horizontal padding between plots in point units.")

multi_parser.add_argument(
    "--vpad",
    nargs="?",
    default=100,
    type=float,
    help="Vertical padding between plots in point units.")

# Different GMT versions
_gmt_five = StrictVersion('5.0')
_gmt_five_two = StrictVersion('5.2')

# python 2/3 compatibility
from six import string_types


class GMT(object):
    def __init__(self, out, debug=False, config=None, **common_flags):
        
        # no margins by default
        self.left, self.right, self.top, self.bottom = 0.0, 0.0, 0.0, 0.0
        
        self.out = out
        
        # get GMT version
        self.version = get_version()
        
        if self.version > _gmt_five:
            self.is_gmt5 = True
        else:
            self.is_gmt5 = False
        
        self.config = _gmt_defaults
        
        if config is not None:
            # convert keys to uppercase
            config = {key.upper(): value for key, value in config.items()}
            self.config.update(config)
            
        if self.config["PS_PAGE_ORIENTATION"] == "portrait":
            self.is_portrait = True
        elif self.config["PS_PAGE_ORIENTATION"] == "landscape":
            self.is_portrait = False
        else:
            raise ValueError('PS_PAGE_ORIENTATION should be either '
                             '"portrait" or "landscape"')
        
        # get paper width and height
        paper = self.get_config("ps_media")

        if paper.startswith("a") or paper.startswith("b"):
            paper = paper.upper()

        if self.is_portrait:
            self.width, self.height = _gmt_paper_sizes[paper]
        else:
            self.height, self.width = _gmt_paper_sizes[paper]
            
        with open("gmt.conf", "w") as f:
            f.write(_gmt_defaults_header)
            f.write("\n".join("{} = {}".format(key.upper(), value)
                              for key, value in self.config.items()))
        
        # list of lists that contains the gmt commands, its arguments, input
        # data and the filename where the command outputs will be written
        self.commands = []
        self.debug = debug
        
        # if we have common flags parse them
        if len(common_flags) > 0:
            self.common = " ".join(["-{}{}".format(key, proc_flag(flag))
                                    for key, flag in common_flags.items()])
        else:
            self.common = ""
        
    def __del__(self):
        commands = self.commands
        
        # indices of plotter functions
        idx = [ii for ii, Cmd in enumerate(commands) if Cmd[0] in _plotters]
        
        # add -K, -P and -O flags to plotter functions
        if len(idx) > 1:
            for ii in idx[:-1]:
                commands[ii][1] += " -K"

            for ii in idx[1:]:
                commands[ii][1] += " -O"
            
        if self.is_portrait:
            for ii in idx:
                commands[ii][1] += " -P"
        
        if self.is_gmt5:
            commands = [["gmt " + Cmd[0], Cmd[1], Cmd[2], Cmd[3]]
                        for Cmd in commands]
        
        if self.debug:
            # print commands for debugging
            print("\n".join(" ".join(elem for elem in Cmd[0:2])
                                          for Cmd in commands))
        
        # gather all the outputfiles and remove the ones that already exist
        outfiles = set(Cmd[3] for Cmd in commands
                       if Cmd[3] is not None and pth.isfile(Cmd[3]))
        
        for out in outfiles:
            os.remove(out)
        
        # execute gmt commands and write their output to the specified files
        for Cmd in commands:
            execute_gmt_cmd(Cmd)
        
        # Cleanup
        if pth.isfile("gmt.history"): os.remove("gmt.history")
        os.remove("gmt.conf")

    def raster(self, out, **kwargs):
        
        dpi           = float(kwargs.pop("dpi", 200))
        gray          = bool(kwargs.pop("gray", False))
        portrait      = bool(kwargs.pop("portrait", False))
        with_pagesize = bool(kwargs.pop("with_pagesize", False))
        multi_page    = bool(kwargs.pop("multi_page", False))
        transparent   = bool(kwargs.pop("transparent", False))
        
        name, ext = pth.splitext(out)
        
        if self.is_five:
            Cmd = "gmt ps2raster"
        else:
            Cmd = "ps2raster"
        
        # extension code
        Ext = _ps2raster[ext]
        
        # handle extra options
        
        if with_pagesize:
            if ext == ".eps":
                Ext = Ext.upper()
            else:
                raise ValueError("with_pagesize only available for EPS files.")
            
        if multi_page:
            if ext == ".pdf":
                Ext = Ext.upper()
            else:
                raise ValueError("multi_page only available for PDF files.")
        
        if transparent:
            if ext == ".png":
                Ext = Ext.upper()
            else:
                raise ValueError("transparent only available for PNG files.")

        
        Cmd = "{} {} -T{} -E{} -F{}"\
              .format(Cmd, self.out, Ext, dpi, name)
        
        if gray:
            Cmd += " -I"
        
        if portrait:
            Cmd += " -P"
        
        cmd(Cmd)
    
    def _gmtcmd(self, gmt_exec, data=None, byte_swap=False, outfile=None, **flags):
        
        if data is not None:
            if isinstance(data, string_types) and pth.isfile(data):
            # data is a path to a file
                gmt_flags = "{} ".format(data)
                data = None
            elif isinstance(data, list) or isinstance(data, tuple):
                data = ("\n".join(elem for elem in data)).encode()
                gmt_flags = ""
            else:
                raise ValueError("`data` should be a path to an existing file!")
        else:
            gmt_flags = ""
        
        # if we have flags parse them
        if len(flags) > 0:
            gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flag))
                                   for key, flag in flags.items()])
        
        # if we have common flags add them
        if self.common is not None:
            gmt_flags += " " + self.common
        
        if outfile is not None:
            self.commands.append([gmt_exec, gmt_flags, data, outfile])
        else:
            self.commands.append([gmt_exec, gmt_flags, data, self.out])
    
    def __getattr__(self, command, *args, **kwargs):
        def f(*args, **kwargs):
            self._gmtcmd(command, *args, **kwargs)
        return f
    
    def reform(self, **common_flags):
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

    def get_config(self, param):
        return self.config[param.upper()]
    
    def get_width(self):
        
        if self.is_gmt5:
            Cmd = "gmt mapproject {} -Dp".format(self.common)
            version = get_version()
        else:
            Cmd = "mapproject {} -Dp".format(self.common)
        
        if not self.version >= _gmt_five_two:
            # before version 5.2
            Cmd += " {} -V".format(os.devnull)
            out = [line for line in cmd(Cmd, ret=True).split("\n")
                   if "Transform" in line]
            return float(out[0].split("/")[4])
        else:
            Cmd += " -Ww"
            return float(cmd(Cmd, ret=True))

    def get_height(self):
        
        if self.is_gmt5:
            Cmd = "gmt mapproject {} -Dp".format(self.common)
            version = get_version()
        else:
            Cmd = "mapproject {} -Dp".format(self.common)
        
        if not self.version >= _gmt_five_two:
            # before version 5.2
            Cmd += " {} -V".format(os.devnull)
            out = [line for line in cmd(Cmd, ret=True).split("\n")
                   if "Transform" in line]
            return float(out[0].split("/")[6].split()[0])
        else:
            Cmd += " -Wh"
            return float(cmd(Cmd, ret=True))
        
    def get_common_flag(self, flag):
        return self.common.split(flag)[1].split()[0]
    
    def multiplot(self, nplots, proj, nrows=None, **kwargs):
        """
             |       top           |    
        -----+---------------------+----
          l  |                     |  r
          e  |                     |  i
          f  |                     |  g
          t  |                     |  h
             |                     |  t
        -----+---------------------+----
             |       bottom        |    
        """
        xpad  = float(kwargs.pop("hpad", 55))
        ypad  = float(kwargs.pop("hpad", 75))
        top   = float(kwargs.pop("top", 0))
        left  = float(kwargs.pop("left", 0))
        right = float(kwargs.pop("right", 0))
        
        if nrows is None:
            nrows = ceil(sqrt(nplots) - 1)
            nrows = max([1, nrows])
        
        self.left, self.right, self.top  = left, top, right
        
        ncols = ceil(nplots / nrows)
        
        width, height = self.width, self.height
        
        # width available for plotting
        awidth = width - (left + right)
        
        # width of a single plot
        pwidth  = float(awidth - (ncols - 1) * xpad) / ncols
        
        self.common += " -J{}{}p".format(proj, pwidth)
        
        # height of a single plot
        pheight = self.get_height()
        
        # calculate psbasemap shifts in x and y directions
        x = (left + ii * (pwidth + xpad) for jj in range(nrows)
                                         for ii in range(ncols))
        
        y = (height - top - ii * (pheight + ypad)
             for ii in range(1, nrows + 1)
             for jj in range(ncols))
        
        # residual margin left at the bottom
        self.bottom = height - top - nrows * pheight
        
        return tuple(x), tuple(y)
    
    def scale_pos(self, mode, offset=100, flong=0.8, fshort=0.2):
        left, right, top, bottom = self.left, self.right, self.top, self.bottom
        
        width, height = self.width, self.height
        
        if mode == "vertical" or mode == "v":
            x = width - left - offset
            y = float(height) / 2
            
            # fraction of space available
            width  = fshort * left
            length = flong * height
            hor = ""
        elif mode == "horizontal" or mode == "h":
            x = float(self.width) / 2
            y = bottom - offset
            
            # fraction of space available
            length  = flong * width
            width   = fshort * bottom
            hor = "h"
        else:
            raise ValueError('mode should be either: "vertical", "horizontal", '
                             '"v" or "h", not "{}"'.format(mode))
        
        return str(x) + "p", str(y) + "p", str(length) + "p",\
               str(width) + "p" +  hor
    
    def colorbar(self, mode="v", offset=100, flong=0.8, fshort=0.2, **flags):
        
        xx, yy, length, width = self.scale_pos(mode, offset=offset,
                                               flong=flong, fshort=fshort)
    
        self.psscale(D=(0.0, 0.0, length, width), Xf=xx, Yf=yy, **flags)

    # end GMT
    
def get_version():
    """ Get the version of the installed GMT as a Strict Version object. """
    return StrictVersion(cmd("gmt --version", ret=True).strip())

def get_paper_size(paper, is_portrait=False):
    """ Get paper width and height. """

    if paper.startswith("a") or paper.startswith("b"):
        paper = paper.upper()

    if is_portrait:
        width, height = _gmt_paper_sizes[paper]
    else:
        height, width = _gmt_paper_sizes[paper]
    
    return width, height
    
def proc_flag(flag):
    """ Parse GMT flags. """
    if isinstance(flag, bool) and flag:
        return ""
    elif hasattr(flag, "__iter__") and not isinstance(flag, string_types):
        return "/".join(str(elem) for elem in flag)
    elif flag is None:
        return ""
    else:
        return flag

def execute_gmt_cmd(Cmd, ret_out=False):
    # join command and flags
    gmt_cmd = Cmd[0] + " " + Cmd[1]
    
    try:
        cmd_out = subp.check_output(split(gmt_cmd), input=Cmd[2],
                                   stderr=subp.STDOUT)
    except subp.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(gmt_cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if Cmd[3] is not None:
        with open(Cmd[3], "ab") as f:
            f.write(cmd_out)
    
    if ret_out:
        return cmd_out

def cmd(Cmd, ret=False):
    """
    Execute terminal command defined by `cmd`, optionally return the
    output of the executed command if `ret` is set to True.
    """
    try:
        cmd_out = subp.check_output(split(Cmd), stderr=subp.STDOUT)
    except subp.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(Cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if ret:
        # filter out gmt messages
        return " ".join(elem for elem in cmd_out.decode().split("\n")
                        if not elem.startswith("gmt:"))

dem_dtypes = {
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

_gmt_defaults_header = \
r'''#
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
"FONT_TITLE": "16p,Helvetica,black",
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
"GMT_HISTORY": "false",
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
"MAP_FRAME_AXES": "WSenZ",
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
"PS_PAGE_ORIENTATION": "landscape",
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

# paper width and height in points
_gmt_paper_sizes = {
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

def shunt(tokens):
    try:
        cmd_out = subp.check_output("shunt2.sh", input=tokens.encode(),
                                   stderr=subp.STDOUT).decode
    except subp.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(gmt_cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    cmd_out = cmd_out.replace("+", "ADD")
    cmd_out = cmd_out.replace("-", "SUB")
    cmd_out = cmd_out.replace("/", "DIV")
    cmd_out = cmd_out.replace("*", "MULT")
    
    return sub(r"FUNARG*INVOKE", "", cmd_out)

# **********************************************
# * Infix to Reverse Polish Notation converter *
# **********************************************

# from http://andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm/

# Associativity constants for operators
LEFT_ASSOC = 0
RIGHT_ASSOC = 1
FUN = 2

# Supported operators
OPERATORS = {
    # A
    "ABS"       : (5, FUN),
    "ACOS"      : (5, FUN),
    "ACOSH"     : (5, FUN),
    "ACSC"      : (5, FUN),
    "ACOT"      : (5, FUN),
    "ADD"       : (0, LEFT_ASSOC),
    "AND"       : (5, LEFT_ASSOC),
    "ASEC"      : (5, FUN),
    "ASIN"      : (5, FUN),
    "ASINH"     : (5, FUN),
    "ATAN"      : (5, FUN),
    "ATAN2"     : (5, FUN),
    "ATANH"     : (5, FUN),
    # B
    "BEI"       : (5, FUN),
    "BER"       : (5, FUN),
    "BITAND"    : (5, LEFT_ASSOC),
    "BITNOT"    : (5, RIGHT_ASSOC),
    "BITOR"     : (5, LEFT_ASSOC),
    "BITRIGHT"  : (5, LEFT_ASSOC),
    "BITTEST"   : (5, LEFT_ASSOC),
    "BITXOR"    : (5, LEFT_ASSOC),
    # C
    "CEIL"      : (5, FUN),
    "CHICRIT"   : (5, FUN),
    "CHIDIST"   : (5, FUN),
    "COL"       : (5, RIGHT_ASSOC),
    "CORRCOEFF" : (5, LEFT_ASSOC),
    "COS"       : (5, FUN),
    "COSD"      : (5, FUN),
    "COSH"      : (5, FUN),
    "COT"       : (5, FUN),
    "COTD"      : (5, FUN),
    "CSC"       : (5, FUN),
    "CSCD"      : (5, FUN),
    "CPOISS"    : (5, LEFT_ASSOC),
    # D
    "DDT"       : (5, FUN),
    "D2DT2"     : (5, FUN),
    "D2R"       : (5, FUN),
    "DILOG"     : (5, FUN),
    "DIFF"      : (5, FUN),
    "DIV"       : (5, LEFT_ASSOC),
    "DUP"       : (5, RIGHT_ASSOC),
    # E
    "ERF"       : (5, FUN),
    "ERFC"      : (5, FUN),
    "ERFINV"    : (5, FUN),
    "EQ"        : (5, LEFT_ASSOC),
    "EXCH"      : (5, LEFT_ASSOC),
    "EXP"       : (5, FUN),
    # F
    "FACT"      : (5, FUN),
    "FCRIT"     : (5, FUN),
    "FDIST"     : (5, FUN),
    "FLIPUD"    : (5, RIGHT_ASSOC),
    "FLOOR"     : (5, FUN),
    "FMOD"      : (5, FUN),
    # G
    "GE"        : (5, LEFT_ASSOC),
    "GT"        : (5, LEFT_ASSOC),
    # H
    "HYPOT"     : (5, FUN),
    # I
    "I0"        : (5, FUN),
    "I1"        : (5, FUN),
    "IFELSE"    : (5, LEFT_ASSOC),
    "IN"        : (5, FUN),
    "INRANGE"   : (5, LEFT_ASSOC),
    "INT"       : (5, FUN),
    "INV"       : (5, RIGHT_ASSOC),
    "ISFINITE"  : (5, RIGHT_ASSOC),
    "ISNAN"     : (5, RIGHT_ASSOC),
    # J
    "J0"        : (5, FUN),
    "J1"        : (5, FUN),
    "JN"        : (5, LEFT_ASSOC),
    # K
    "K0"        : (5, FUN),
    "K1"        : (5, FUN),
    "KN"        : (5, LEFT_ASSOC),
    "KEI"       : (5, FUN),
    "KER"       : (5, FUN),
    "KURT"      : (5, FUN),
    # L
    "LE"        : (5, LEFT_ASSOC),
    "LMSSCL"    : (5, FUN),
    "LOG"       : (5, FUN),
    "LOG10"     : (5, FUN),
    "LOG1P"     : (5, FUN),
    "LOG2"      : (5, FUN),
    "LOWER"     : (5, FUN),
    "LRAND"     : (5, LEFT_ASSOC),
    "LSQFIT"    : (5, RIGHT_ASSOC),
    "LT"        : (5, LEFT_ASSOC),
    # M
    "MAD"        : (5, FUN),
    "MAX"        : (5, LEFT_ASSOC),
    # Fury Road
    "MEAN"       : (5, FUN),
    "MED"        : (5, FUN),
    "MIN"        : (5, LEFT_ASSOC),
    "MOD"        : (5, LEFT_ASSOC),
    "MODE"       : (5, FUN),
    "MUL"        : (5, LEFT_ASSOC),
    # N
    "NAN"        : (5, FUN),
    "NEG"        : (5, RIGHT_ASSOC),
    "NEQ"        : (5, FUN),
    "NORM"       : (5, FUN),
    "NOT"        : (5, RIGHT_ASSOC),
    "NRAND"      : (5, FUN),
    # O
    "OR"         : (5, LEFT_ASSOC),
    # P
    "PLM"        : (5, FUN),
    "PLMg"       : (5, FUN),
    "POP"        : (5, FUN),
    "POW"        : (10, FUN),
    "PQUANT"     : (5, FUN),
    "PSI"        : (5, FUN),
    "PV"         : (5, FUN),
    # Q
    "QV"         : (5, FUN),
    # R
    "R2"         : (5, FUN),
    "R2D"        : (5, FUN),
    "RAND"       : (5, FUN),
    "RINT"       : (5, FUN),
    "ROTT"       : (5, FUN),
    # S
    "SEC"        : (5, FUN),
    "SECD"       : (5, FUN),
    "SIGN"       : (5, FUN),
    "SIN"        : (5, FUN),
    "SINC"       : (5, FUN),
    "SIND"       : (5, FUN),
    "SINH"       : (5, FUN),
    "SKEW"       : (5, FUN),
    "SQR"        : (5, FUN),
    "SQRT"       : (5, FUN),
    "STD"        : (5, FUN),
    "STEP"       : (5, FUN),
    "STEPT"      : (5, FUN),
    "SUB"        : (0, LEFT_ASSOC),
    "SUM"        : (5, FUN),
    # T
    "TAN"        : (5, FUN),
    "TAND"       : (5, FUN),
    "TANH"       : (5, FUN),
    "TAPER"      : (5, FUN),
    "TN"         : (5, FUN),
    "TCRIT"      : (5, FUN),
    "TDIST"      : (5, FUN),
    # U
    "UPPER"      : (5, FUN),
    # X
    "XOR"        : (5, LEFT_ASSOC),
    # Y
    "Y0"         : (5, FUN),
    "Y1"         : (5, FUN),
    "YN"         : (5, FUN),
    # Z
    "ZCRIT"      : (5, FUN),
    "ZDIST"      : (5, FUN),
    # R
    "ROOTS"      : (5, FUN)
}

# Test if a certain token is operator
def is_operator(token):
    return token in OPERATORS.keys() and OPERATORS[token][1] != 2

# Test the associativity type of a certain token
def is_associative(token, assoc):
    if not is_operator(token):
        raise ValueError("Invalid token: {}".format(token))
    return OPERATORS[token][1] == assoc

# Compare the precedence of two tokens
def cmp_precedence(token1, token2):
    if not is_operator(token1) or not is_operator(token2):
        raise ValueError('Invalid tokens: {} {}'.format(token1, token2))
    return OPERATORS[token1][0] - OPERATORS[token2][0]

def is_fun(token):
    return token in OPERATORS.keys() and OPERATORS[token][1] == 2

# Transforms an infix expression to RPN
def infix2RPN(tokens):
    out = []
    stack = []
    
    print(tokens)
    
    # For all the input tokens [S1] read the next token [S2]
    for token in tokens:
        isfun = is_fun(token)
        isop  = is_operator(token)
        
        if not isop and not isfun:
            out.append(token)
        if isfun:
            stack.append(token)
        if isop:
            # If token is an operator (x) [S3]
            while len(stack) != 0 and is_fun(stack[-1]):
                # [S4]
                print(stack)
                if (is_associative(token, LEFT_ASSOC) \
                    and cmp_precedence(token, stack[-1]) <= 0) or \
                    (is_associative(token, RIGHT_ASSOC) \
                    and cmp_precedence(token, stack[-1]) < 0):
                    # [S5] [S6]
                    out.append(stack.pop())
                    continue
                break
            # [S7]
            stack.append(token)
        elif token == '(':
            stack.append(token) # [S8]
        elif token == ')':
            # [S9]
            while len(stack) != 0 and stack[-1] != '(':
                out.append(stack.pop()) # [S10]
            stack.pop() # [S11]
        else:
            out.append(token) # [S12]
    while len(stack) != 0:
        # [S13]
        out.append(stack.pop())
    return out
