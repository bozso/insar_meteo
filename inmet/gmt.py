import os
import os.path as pth
import subprocess as sub
from shlex import split
from math import ceil, sqrt
from distutils.version import StrictVersion

import glob

_gmt_five = StrictVersion('5.0')
_gmt_five_two = StrictVersion('5.2')

# python 2/3 compatibility
from six import string_types

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
            self.common = None
        
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

    def raster(self, out, dpi=200, gray=False, portrait=False,
               with_pagesize=False, multi_page=False, transparent=False):
        
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

        
        outdir = pth.dirname(out)
        
        if outdir == "":
            outdir = "."
        
        Cmd = "{} {} -T{} -E{} -D{} -F{}"\
              .format(Cmd, self.out, Ext, dpi, outdir, pth.basename(name))
        
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
            version = cmd("gmt --version", ret_out=True)
        else:
            Cmd = "mapproject {} -Dp".format(self.common)
        
        if not self.version >= _gmt_five_two:
            # before version 5.2
            Cmd += " {} -V".format(os.devnull)
            out = [line for line in cmd(Cmd, ret_out=True).decode().split("\n")
                   if "Transform" in line]
            return float(out[0].split("/")[4])
        else:
            Cmd += " -Ww"
            return float(cmd(Cmd, ret_out=True))

    def get_height(self):
        
        if self.is_gmt5:
            Cmd = "gmt mapproject {} -Dp".format(self.common)
            version = cmd("gmt --version", ret_out=True)
        else:
            Cmd = "mapproject {} -Dp".format(self.common)
        
        if not self.version >= _gmt_five_two:
            # before version 5.2
            Cmd += " {} -V".format(os.devnull)
            out = [line for line in cmd(Cmd, ret_out=True).decode().split("\n")
                   if "Transform" in line]
            return float(out[0].split("/")[6].split()[0])
        else:
            Cmd += " -Wh"
            return float(cmd(Cmd, ret_out=True))
        
    def get_common_flag(self, flag):
        return self.common.split(flag)[1].split()[0]
    
    def multiplot(self, nplots, proj, nrows=None, top=0, left=25, right=50,
                  x_pad=55, y_pad=75):
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
        if nrows is None:
            nrows = ceil(sqrt(nplots) - 1)
            nrows = max([1, nrows])
        
        self.left, self.right, self.top  = left, top, right
        
        ncols = ceil(nplots / nrows)
        
        width, height = self.width, self.height
        
        # width available for plotting
        awidth = width - (left + right)
        
        # width of a single plot
        pwidth  = float(awidth - (ncols - 1) * x_pad) / ncols
        
        self.common += " -J{}{}p".format(proj, pwidth)
        
        # height of a single plot
        pheight = self.get_height()
        
        # calculate psbasemap shifts in x and y directions
        x = (left + ii * (pwidth + x_pad) for jj in range(nrows)
                                          for ii in range(ncols))
        
        y = (height - top - ii * (pheight + y_pad)
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
    
def info(data, **flags):
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
    
    if get_version() > _gmt_five:
        Cmd = "gmt info " + gmt_flags
    else:
        Cmd = "gmtinfo " + gmt_flags
    
    return cmd(Cmd, ret_out=True).decode()

def get_ranges(data, binary=None, xy_add=None, z_add=None):
    
    if binary is not None:
        info_str = info(data, bi=binary, C=True).split()
    else:
        info_str = info(data, C=True).split()
    
    ranges = tuple(float(data) for data in info_str)
    
    if xy_add is not None:
        X = (ranges[1] - ranges[0]) * xy_add
        Y = (ranges[3] - ranges[2]) * xy_add
        xy_range = (ranges[0] - xy_add, ranges[1] + xy_add,
                    ranges[2] - xy_add, ranges[3] + xy_add)
    else:
        xy_range = ranges[0:4]

    non_xy = ranges[4:]
    
    if z_add is not None:
        min_z, max_z = min(non_xy), max(non_xy)
        Z = (max_z - min_z) * z_add
        z_range = (min_z - z_add, max_z + z_add)
    else:
        z_range = (min(non_xy), max(non_xy))
        
    return xy_range, z_range

def get_version():
    """ Get the version of the installed GMT as a Strict Version object. """
    return StrictVersion(cmd("gmt --version", ret_out=True).decode().strip())

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
        cmd_out = sub.check_output(split(gmt_cmd), input=Cmd[2],
                                   stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(gmt_cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if Cmd[3] is not None:
        with open(Cmd[3], "ab") as f:
            f.write(cmd_out)
    
    if ret_out:
        return cmd_out

def cmd(Cmd, ret_out=False):
    """
    Execute terminal command defined by `cmd`, optionally return the
    output of the executed command if `ret_out` is set to True.
    """
    try:
        cmd_out = sub.check_output(split(Cmd), stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(Cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if ret_out:
        return cmd_out

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

def make_ncfile(self, header_path, ncfile, endian="small", gmt5=True):
    fmt = get_par("SAM_IN_FORMAT", header_path)
    dempath = get_par("SAM_IN_DEM", header_path)
    nodata = get_par("SAM_IN_NODATA", header_path)
    
    rows, cols = get_par("SAM_IN_SIZE", header_path).split()
    delta_lat, delta_lon = get_par("SAM_IN_DELTA", header_path).split()
    origin_lat, origin_lon = get_par("SAM_IN_UL", header_path).split()
    
    xmin = float(origin_lon)
    xmax = xmin + float(cols) * float(delta_lon)

    ymax = float(origin_lat)
    ymin = ymax - float(rows) * float(delta_lat)
    
    lonlat_range = "{}/{}/{}/{}".format(xmin ,xmax, ymin, ymax)
    
    increments = "{}/{}".format(self.delta_lon, self.delta_lat)
    
    Cmd = "xyz2grd {infile} -ZTL{dtype} -R{ll_range} -I{inc} -r -G{nc}"\
          .format(infile=dempath, dtype=dem_dtypes[fmt], ll_range=lonlat_range,
                  inc=increments, nc=ncfile)
    
    if get_version() > _gmt_five:
        Cmd = "gmt " + Cmd
    
    cmd(Cmd)
    
def plot_scatter(scatter_file, ncols, outfile, proj="M", idx=None, config=None,
                 cbar_config={}, axis_config={}, colorscale="drywet",
                 right=100, top=0, left=50, tryaxis=False,
                 titles=None, xy_range=None, z_range=None,
                 x_axis="a0.5g0.25f0.25", y_axis="a0.25g0.25f0.25",
                 xy_add=0.05, z_add=0.1, **kwargs):

    name, ext = pth.splitext(outfile)
    
    if ext != ".ps":
        ps_file = name + ".ps"
    else:
        ps_file = outfile
    
    # 2 additional coloumns for coordinates
    bindef = "{}d".format(ncols + 2)
    
    if xy_range is None or z_range is None:
        _xy_range, _z_range = get_ranges(data=scatter_file, binary=bindef,
                                          xy_add=xy_add, z_add=z_add)
    
    if xy_range is None:
        xy_range = _xy_range
    if z_range is None:
        z_range = _z_range
    
    if idx is None:
        idx = range(ncols)
    
    if titles is None:
        titles = range(1, ncols + 1)
    
    gmt = GMT(ps_file, R=xy_range, config=config)
    x, y = gmt.multiplot(len(idx), proj, right=right, top=top, left=left)
    
    gmt.makecpt("tmp.cpt", C=colorscale, Z=True, T=z_range)
    
    for ii in idx:
        input_format = "0,1,{}".format(ii + 2)
        gmt.psbasemap(Xf="{}p".format(x[ii]), Yf="{}p".format(y[ii]),
                      B="WSen+t{}".format(titles[ii]), Bx=x_axis, By=y_axis)
        
        # do not plot the scatter points yet just see the placement of
        # basemaps
        if not tryaxis:
            gmt.psxy(data=scatter_file, i=input_format, bi=bindef,
                     S="c0.025c", C="tmp.cpt")
        
    gmt.colorbar(mode=kwargs.pop("mode", "v"), offset=kwargs.pop("offset", 10),
                 B=kwargs.pop("label", ""), C="tmp.cpt",)
    
    if ext != ".ps":
        gmt.raster(outfile, **kwargs)
        os.remove(ps_file)
    
    os.remove("tmp.cpt")
    
    del gmt

def hist(data, ps_file, binwidth=0.1, config=None, binary=None, 
         left=50, right=25, top=25, bottom=50, **flags):
    
    ranges = tuple(float(elem)
                   for elem in info(data, bi=binary, C=True).split())
    
    min_r, max_r = min(ranges), max(ranges)
    binwidth = (max_r - min_r) * binwidth
    
    width, height = get_paper_size(config.pop("PS_MEDIA", "A4"))
    
    proj="X{}p/{}p".format(width - left - right, height - top - bottom)
    
    gmt = GMT(ps_file, config=config, R=(min_r, max_r, 0.0, 100.0), J=proj)
    
    gmt.psbasemap(Bx="a{}".format(round(binwidth)), By=5,
                  Xf=str(left) + "p", Yf=str(bottom) + "p")
    
    gmt.pshistogram(data=data, W=binwidth, G="blue", bi=binary, Z=1)

    del gmt

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

# **********************************************
# * Infix to Reverse Polish Notation converter *
# **********************************************

# from http://andreinc.net/2010/10/05/converting-infix-to-rpn-shunting-yard-algorithm/

# Associativity constants for operators
LEFT_ASSOC = 0
RIGHT_ASSOC = 1

# Supported operators
OPERATORS = {
    # A
    "ABS"     : (5, RIGHT_ASSOC),
    "ACOS"    : (5, RIGHT_ASSOC),
    "ACOSH"    : (5, RIGHT_ASSOC),
    "ACSC"    : (5, RIGHT_ASSOC),
    "ACOT"    : (5, RIGHT_ASSOC),
    "ADD"     : (0, LEFT_ASSOC),
    "AND"     : (0, LEFT_ASSOC),
    "ASEC"    : (5, RIGHT_ASSOC),
    "ASIN"    : (5, RIGHT_ASSOC),
    "ASINH"   : (5, RIGHT_ASSOC),
    "ATAN"    : (5, RIGHT_ASSOC),
    "ATAN2"    : (5, LEFT_ASSOC),
    "ATANH"    : (5, RIGHT_ASSOC),
    # B
    "SUB"     : (0, LEFT_ASSOC),
    "MUL"     : (5, LEFT_ASSOC),
    "DIV"     : (5, LEFT_ASSOC),
    "%"       : (5, LEFT_ASSOC),
    "POW"     : (10, RIGHT_ASSOC)
}

# Test if a certain token is operator
def isOperator(token):
    return token in OPERATORS.keys()

# Test the associativity type of a certain token
def isAssociative(token, assoc):
    if not isOperator(token):
        raise ValueError('Invalid token: %s' % token)
    return OPERATORS[token][1] == assoc

# Compare the precedence of two tokens
def cmpPrecedence(token1, token2):
    if not isOperator(token1) or not isOperator(token2):
        raise ValueError('Invalid tokens: %s %s' % (token1, token2))
    return OPERATORS[token1][0] - OPERATORS[token2][0]

# Transforms an infix expression to RPN
def infix2RPN(tokens):
    out = []
    stack = []
    
    # For all the input tokens [S1] read the next token [S2]
    for token in tokens:
        if isOperator(token):
            
            # If token is an operator (x) [S3]
            while len(stack) != 0 and isOperator(stack[-1]):
                
                # [S4]
                if (isAssociative(token, LEFT_ASSOC) \
                    and cmpPrecedence(token, stack[-1]) <= 0) or \
                    (isAssociative(token, RIGHT_ASSOC) \
                    and cmpPrecedence(token, stack[-1]) < 0):
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
