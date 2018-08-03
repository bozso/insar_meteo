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

from sys import platform
from os import linesep, popen

if platform == "mac":

    import Required_Suite
    import aetools

    class _GNUPLOT(aetools.TalkTo,
                   Required_Suite.Required_Suite,
                   gnuplot_Suites.gnuplot_Suite,
                   gnuplot_Suites.odds_and_ends,
                   gnuplot_Suites.Standard_Suite,
                   gnuplot_Suites.Miscellaneous_Events):
        """Start a gnuplot program and emulate a pipe to it."""
    
        def __init__(self):
            aetools.TalkTo.__init__(self, '{GP}', start=1)


def _test_persist_unix_macosx(gnuplot_command, recognizes_persist):
    """
    Determine whether gnuplot recognizes the option '-persist'.
    If the configuration variable 'recognizes_persist' is set (i.e.,
    to something other than 'None'), return that value.  Otherwise,
    try to determine whether the installed version of gnuplot
    recognizes the -persist option.  (If it doesn't, it should emit an
    error message with '-persist' in the first line.)  Then set
    'recognizes_persist' accordingly for future reference.
    """

    if recognizes_persist is None:
        g = popen("echo | %s -persist 2>&1" % gnuplot_command, "r")
        response = g.readlines()
        g.close()
        recognizes_persist = not (response and "-persist" in response[0])
    return recognizes_persist


class GnuplotProcessUnix(object):
    """
    Unsophisticated interface to a running gnuplot program.
    This represents a running gnuplot program and the means to
    communicate with it at a primitive level (i.e., pass it commands
    or data).  When the object is destroyed, the gnuplot program exits
    (unless the 'persist' option was set).  The communication is
    one-way; gnuplot's text output just goes to stdout with no attempt
    to check it for error messages.
    Members:
        'gnuplot' -- the pipe to the gnuplot command.
    Methods:
        '__init__' -- start up the program.
        '__call__' -- pass an arbitrary string to the gnuplot program,
            followed by a newline.
        'write' -- pass an arbitrary string to the gnuplot program.
        'flush' -- cause pending output to be written immediately.
        'close' -- close the connection to gnuplot.
    """

    gnuplot_command = "gnuplot"
    recognizes_persist =  None
    prefer_persist =  0
    recognizes_binary_splot =  1
    prefer_inline_data =  0
    default_term =  "x11"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=None):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
          'persist=1' -- start gnuplot with the '-persist' option,
              (which leaves the plot window on the screen even after
              the gnuplot program ends, and creates a new plot window
              each time the terminal type is set to 'x11').  This
              option is not available on older versions of gnuplot.
        """

        if persist is None:
            persist = self.prefer_persist
        if persist:
            if not _test_persist_unix_macosx(self.gnuplot_command,
                                             self.recognizes_persist):
                raise OptionError("-persist does not seem to be supported "
                                  "by your version of gnuplot!")
            self.gnuplot = popen("%s -persist".format(self.gnuplot_command),
                                 "w")
        else:
            self.gnuplot = popen(self.gnuplot_command, "w")
        
        # forward write and flush methods:
        self.write = self.gnuplot.write
        self.flush = self.gnuplot.flush

    def close(self):
        if self.gnuplot is not None:
            self.gnuplot.close()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def __call__(self, s):
        """Send a command string to gnuplot, followed by newline."""

        self.write(s + "\n")
        self.flush()

class GnuplotProcessWin32(object):
    """
    Unsophisticated interface to a running gnuplot program.
    See gp_unix.py for usage information.
    """

    gnuplot_command = r"pgnuplot.exe"
    recognizes_persist = 0
    recognizes_binary_splot =  1
    prefer_inline_data =  0
    default_term =  "windows"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=0):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
            'persist' -- the '-persist' option is not supported under
                Windows so this argument must be zero.
        """

        if persist:
            raise OptionError("-persist is not supported under Windows!")

        self.gnuplot = popen(self.gnuplot_command, "w")

        # forward write and flush methods:
        self.write = self.gnuplot.write
        self.flush = self.gnuplot.flush

    def close(self):
        if self.gnuplot is not None:
            self.gnuplot.close()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def __call__(self, s):
        """Send a command string to gnuplot, followed by newline."""

        self.write(s + "\n")
        self.flush()

    def test_persist(self):
        return 0

class GnuplotProcessMacOSX(object):
    """
    Unsophisticated interface to a running gnuplot program.
    This represents a running gnuplot program and the means to
    communicate with it at a primitive level (i.e., pass it commands
    or data).  When the object is destroyed, the gnuplot program exits
    (unless the 'persist' option was set).  The communication is
    one-way; gnuplot's text output just goes to stdout with no attempt
    to check it for error messages.
    Members:
        'gnuplot' -- the pipe to the gnuplot command.
    Methods:
        '__init__' -- start up the program.
        '__call__' -- pass an arbitrary string to the gnuplot program,
            followed by a newline.
        'write' -- pass an arbitrary string to the gnuplot program.
        'flush' -- cause pending output to be written immediately.
        'close' -- close the connection to gnuplot.
    """

    gnuplot_command = "gnuplot"
    recognizes_persist =  None
    prefer_persist =  0
    recognizes_binary_splot =  1
    prefer_inline_data =  0
    default_term =  "aqua"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=None):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
          'persist=1' -- start gnuplot with the '-persist' option,
              (which leaves the plot window on the screen even after
              the gnuplot program ends, and creates a new plot window
              each time the terminal type is set to 'x11').  This
              option is not available on older versions of gnuplot.
        """

        if persist is None:
            persist = self.prefer_persist
        if persist:
            if not _test_persist_unix_macosx(self.gnuplot_command,
                                             self.prefer_persist):
                raise OptionError("-persist does not seem to be supported "
                                  "by your version of gnuplot!")
            self.gnuplot = popen("%s -persist" % self.gnuplot_command,
                                 "w")
        else:
            self.gnuplot = popen(self.gnuplot_command, "w")

        # forward write and flush methods:
        self.write = self.gnuplot.write
        self.flush = self.gnuplot.flush

    def close(self):
        if self.gnuplot is not None:
            self.gnuplot.close()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def __call__(self, s):
        """Send a command string to gnuplot, followed by newline."""

        self.write(s + "\n")
        self.flush()


class GnuplotProcessMac(object):
    """
    Unsophisticated interface to a running gnuplot program.
    See gp_unix.GnuplotProcess for usage information.
    """

    recognizes_persist = 0
    recognizes_binary_splot =  1
    prefer_inline_data =  0
    default_term =  "pict"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=0):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
          'persist' -- the '-persist' option is not supported on the
                       Macintosh so this argument must be zero.
        """

        if persist:
            raise OptionError("-persist is not supported on the Macintosh!")

        self.gnuplot = _GNUPLOT()

    def close(self):
        if self.gnuplot is not None:
            self.gnuplot.quit()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def write(self, s):
        """Mac gnuplot apparently requires '\r' to end statements."""

        self.gnuplot.gnuexec(s.replace("\n", linesep))

    def flush(self):
        pass
        
    def __call__(self, s):
        """Send a command string to gnuplot, for immediate execution."""

        # Apple Script doesn't seem to need the trailing '\n'.
        self.write(s)
        self.flush()

    def test_persist(self):
        return 0

class GnuplotProcessJava(object):
    """
    Unsophisticated interface to a running gnuplot program.
    This represents a running gnuplot program and the means to
    communicate with it at a primitive level (i.e., pass it commands
    or data).  When the object is destroyed, the gnuplot program exits
    (unless the 'persist' option was set).  The communication is
    one-way; gnuplot's text output just goes to stdout with no attempt
    to check it for error messages.
    Members:
    Methods:
        '__init__' -- start up the program.
        '__call__' -- pass an arbitrary string to the gnuplot program,
            followed by a newline.
        'write' -- pass an arbitrary string to the gnuplot program.
        'flush' -- cause pending output to be written immediately.
    """
    
    gnuplot_command = "gnuplot"
    recognizes_persist = 1
    prefer_persist =  0
    recognizes_binary_splot =  1
    prefer_inline_data =  0
    default_term =  "x11"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=None):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
          'persist=1' -- start gnuplot with the '-persist' option,
              (which leaves the plot window on the screen even after
              the gnuplot program ends, and creates a new plot window
              each time the terminal type is set to 'x11').  This
              option is not available on older versions of gnuplot.
        """

        if persist is None:
            persist = self.prefer_persist
        
        command = [self.gnuplot_command]
        
        if persist:
            if not self.test_persist():
                raise OptionError("-persist does not seem to be supported "
                                  "by your version of gnuplot!")
            command.append("-persist")

        # This is a kludge: distutils wants to import everything it
        # sees when making a distribution, and if we just call exec()
        # normally that causes a SyntaxError in CPython because "exec"
        # is a keyword.  Therefore, we call the exec() method
        # indirectly.
        #self.process = Runtime.getRuntime().exec(command)
        exec_method = getattr(Runtime.getRuntime(), "exec")
        self.process = exec_method(command)

        self.outprocessor = OutputProcessor(
            "gnuplot standard output processor",
            self.process.getInputStream(), sys.stdout
            )
        self.outprocessor.start()
        self.errprocessor = OutputProcessor(
            "gnuplot standard error processor",
            self.process.getErrorStream(), sys.stderr
            )
        self.errprocessor.start()

        self.gnuplot = self.process.getOutputStream()

    def close(self):
        # ### Does this close the gnuplot process under Java?
        if self.gnuplot is not None:
            self.gnuplot.close()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def write(self, s):
        self.gnuplot.write(s)

    def flush(self):
        self.gnuplot.flush()

    def __call__(self, s):
        """Send a command string to gnuplot, followed by newline."""

        self.write(s + "\n")
        self.flush()

    def test_persist(self):
        """
        Determine whether gnuplot recognizes the option '-persist'.
        """
    
        return self.recognizes_persist


class GnuplotProcessCygwin(object):
    """
    Unsophisticated interface to a running gnuplot program.
    See gp_unix.py for usage information.
    """
    gnuplot_command = "pgnuplot.exe"
    recognizes_persist = 0
    recognizes_binary_splot =  1
    prefer_inline_data =  1
    default_term =  "windows"
    prefer_enhanced_postscript =  1

    def __init__(self, persist=0):
        """
        Start a gnuplot process.
        Create a 'GnuplotProcess' object.  This starts a gnuplot
        program and prepares to write commands to it.
        Keyword arguments:
            'persist' -- the '-persist' option is not supported under
                Windows so this argument must be zero.
        """

        if persist:
            raise OptionError("-persist is not supported under Windows!")

        self.gnuplot = popen(self.gnuplot_command, 'w')

        # forward write and flush methods:
        self.write = self.gnuplot.write
        self.flush = self.gnuplot.flush

    def close(self):
        if self.gnuplot is not None:
            self.gnuplot.close()
            self.gnuplot = None

    def __del__(self):
        self.close()

    def __call__(self, s):
        """Send a command string to gnuplot, followed by newline."""

        self.write(s + "\n")
        self.flush()


if platform == "mac":
    # TODO: from . import gnuplot_Suites

    import Required_Suite
    import aetools

    class _GNUPLOT(aetools.TalkTo,
                   Required_Suite.Required_Suite,
                   gnuplot_Suites.gnuplot_Suite,
                   gnuplot_Suites.odds_and_ends,
                   gnuplot_Suites.Standard_Suite,
                   gnuplot_Suites.Miscellaneous_Events):
        """Start a gnuplot program and emulate a pipe to it."""
    
        def __init__(self):
            aetools.TalkTo.__init__(self, '{GP}', start=1)
    
    gnuplot_proc = GnuplotProcessMac

elif platform == "win32" or platform == "cli":
    try:
        from sys import hexversion
    except ImportError:
        hexversion = 0

    gnuplot_proc = GnuplotProcessWin32

elif platform == "darwin":

    gnuplot_proc = GnuplotProcessMacOSX

elif platform[:4] == "java":

    from java.lang import Runtime

    gnuplot_proc = GnuplotProcessJava

elif platform == "cygwin":

    try:
        from sys import hexversion
    except ImportError:
        hexversion = 0


    gnuplot_proc = GnuplotProcessCygwin

else:
    gnuplot_proc = GnuplotProcessUnix
