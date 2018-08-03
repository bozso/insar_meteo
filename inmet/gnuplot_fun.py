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

from inmet.gnuplot import Gnuplot, linedef

def plot_scatter(data, ncols, out=None, title=None, idx=None, titles=None,
                 palette="rgb", endian="default", portrait=False):
    
    binary = "%double" * (ncols + 2)
    
    if idx is None:
        idx = range(ncols)
    
    # parse titles
    if titles is None:
        titles = range(1, ncols + 1)
    
    gpt = Gnuplot()
    
    # output has to be set before multiplot
    if out is not None:
        gpt.output(out)
    
    #gpt.margins(rmargin=0.1, screen=True)
    
    gpt.multiplot(len(idx), title=title, portrait=portrait)
    
    # gpt.binary("format='{}' endian={}".format(binary, endian))
    
    gpt.xtics(0.4)
    
    # gpt.palette(palette)
    # gpt.colorbar(cbfomat="%4.2f")
    
    #gpt.unset("colorbox")
    
    for ii in idx:
        inp_format = "1:2:{}".format(ii + 3)
        
        plotd = gpt.infile(data, binary=binary, endian=endian,
                           using=inp_format, vith=linedef(pt_type="dot")
                           + " palette")
        gpt.plot(plotd)


# BACKUP

_options_unix = {
    "gnuplot_command" : "gnuplot",
    "recognizes_persist" : None,
    "prefer_persist" : 0,
    "recognizes_binary_splot" : 1,
    "prefer_inline_data" : 0,
    "default_term" : "x11",
    "prefer_enhanced_postscript" : 1
}

_options_win32 = {
    "gnuplot_command" : r"pgnuplot.exe",
    "recognizes_persist" : 0,
    "prefer_persist" : 0,
    "recognizes_binary_splot" : 0,
    "prefer_inline_data" : 0,
    "default_term" : "windows",
    "prefer_enhanced_postscript" : 1
}

_options_macosx = {
    "gnuplot_command" : "gnuplot",
    "recognizes_persist" : None,
    "prefer_persist" : 0,
    "recognizes_binary_splot" : 1,
    "prefer_inline_data" : 0,
    "default_term" : "aqua",
    "prefer_enhanced_postscript" : 1
}

_options_mac = {
    "recognizes_persist" : 0,
    "recognizes_binary_splot" : 1,
    "prefer_inline_data" : 0,
    "default_term" : "pict",
    "prefer_enhanced_postscript" : 1
}

_options_java = {
    "gnuplot_command" : "gnuplot",
    "recognizes_persist" : 1,
    "prefer_persist" : 0,
    "recognizes_binary_splot" : 1,
    "prefer_inline_data" : 0,
    "default_term" : "x11",
    "prefer_enhanced_postscript" : 1
}

_options_cygwin = {
    "gnuplot_command" : "pgnuplot.exe",
    "recognizes_persist" : 0,
    "recognizes_binary_splot" : 1,
    "prefer_inline_data" : 1,
    "default_term" : "windows",
    "prefer_enhanced_postscript" : 1
}


def _unix(persist=None)
    
    if persist is None:
        persist = _options_unix["prefer_persist"]
    if persist:
        if not _test_persist_unix_macosx(_options_unix["gnuplot_command"],
                                         _options_unix["recogizes_persist"]):
            raise OptionError("-persist does not seem to be supported "
                              "by your version of gnuplot!")
        process = popen("%s -persist".format(self.gnuplot_command),
                             "w")
    else:
        process = popen(self.gnuplot_command, "w")
    
    return process, process.write, process.flush

def _win32(persist=False):
    
    if persist:
        raise OptionError("-persist is not supported under Windows!")
    
    process = popen(_options_win32["gnuplot_command"], "w")
    
    return process, process.write, process.flush

def _macosx(persist=None)
    
    if persist is None:
        persist = _options_macosx["prefer_persist"]
    if persist:
        if not _test_persist_unix_macosx(_options_macosx["gnuplot_command"],
                                         _options_macosx["recogizes_persist"]):
            raise OptionError("-persist does not seem to be supported "
                              "by your version of gnuplot!")
        process = popen("%s -persist".format(self.gnuplot_command),
                             "w")
    else:
        process = popen(self.gnuplot_command, "w")
    
    return process, process.write, process.flush

def _mac(persist=0):

    if persist:
        raise OptionError("-persist is not supported on the Macintosh!")

    self.gnuplot = _GNUPLOT()

    def write(self, s):
        """Mac gnuplot apparently requires '\r' to end statements."""

        self.process.gnuexec(s.replace("\n", linesep))
