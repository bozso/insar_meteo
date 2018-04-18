import os
import os.path as pth
import subprocess as sub
import numpy as np

# python 2/3 compatibility
from six import string_types
from shlex import split

# plotting functions that will use the common output filename and the -K
# and -O flags
_plotters = ["grdcontour", "grdimage", "grdvector", "grdview", "psbasemap",
             "psclip", "pscoast", "pscontour", "pshistogram", "psimage",
             "pslegend", "psmask", "psrose", "psscale", "pstext", "pswiggle",
             "psxy", "psxyz", "gmtlogo"]

class GMT(object):
    def __init__(self, out, gmt5=True, portrait=False, **common_flags):
        
        # common output Postscript filename
        self.out = out
        
        # list of sets that contain the gmt commands and its argumants
        # and the filename where the gmt commands output will be written
        self.commands = []
        self.is_gmt5 = gmt5
        
        keys = common_flags.keys()
        
        # if we have common flags parse them
        if len(keys) > 0:
            tmp = ["-{}{}".format(key, proc_flag(common_flags[key]))
                   for key in keys if key not in ["portrait"]]
            self.common = " ".join(tmp)
            
            if portrait:
                self.common += " -P"
            
        else:
            self.common = None

    def __del__(self):
        commands = self.commands
        
        # add -K and -O flags
        if len(self.commands) > 1:
            commands[:-1] = [(cmd[0], cmd[1] + " -K", *cmd[2:])
                             if cmd[0] in _plotters else cmd
                             for cmd in commands[:-1]]
            commands[1:] = [(cmd[0], cmd[1] + " -O", *cmd[2:])
                             if cmd[0] in _plotters else cmd
                             for cmd in commands[1:]]
        if self.is_gmt5:
            commands = [("gmt " + cmd[0], *cmd[1:]) for cmd in commands]
        
        print("\n".join(" ".join(elem for elem in cmd[0:2]) for cmd in commands))
        
        # gather all the outputfiles and remove the ones that already exist
        outfiles = set(cmd[3] for cmd in commands
                       if cmd[3] is not None and pth.isfile(cmd[3]))
        for out in outfiles:
            os.remove(out)
        
        # execute gmt commands and write their output to the specified files
        for cmd in commands:
            execute_cmd(cmd)
        
        del self.commands
        del self.common
        del self.is_gmt5
        del self.out
    
    def _gmtcmd(self, gmt_exec, data=None, palette=None, outfile=None,
                **flags):
        
        keys = flags.keys()
        
        gmt_flags = ""
        
        if data is not None:
            # data is a path to a file
            if isinstance(data, string_types) and pth.isfile(data):
                gmt_flags += " {}".format(data)
                data = None
            # data is a numpy array
            elif isinstance(data, np.ndarray):
                gmt_flags += " -bi{}dw".format(data.shape[1])
                data = data.tobytes()
                
        # if we have flags
        if len(keys) > 0:
            gmt_flags += " ".join(["-{}{}".format(key, proc_flag(flags[key]))
                               for key in keys])
        
        if palette:
            gmt_flags += " -C{}".format(palette)
        
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
        keys = common_flags.keys()
        
        # if we have common flags parse them
        if len(keys) > 0:
            tmp = ["-{}{}".format(key, proc_flag(common_flags[key]))
                   for key in keys if key not in ["portrait"]]
            self.common = " ".join(tmp)
            
            if portrait:
                self.common += " -P"
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

def execute_cmd(cmd, ret_out=False):
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
