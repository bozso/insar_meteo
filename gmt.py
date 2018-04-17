import os
import os.path as pth
import subprocess as sub
import numpy as np

# python 2/3 compatibility
from six import string_types
from shlex import split
from io import BytesIO

# plotting functions that will use the common output filename and the -K
# and -O flags
_plotters = ["grdcontour", "grdimage", "grdvector", "grdview", "psbasemap",
             "psclip", "pscoast", "pscontour", "pshistogram", "psimage",
             "pslegend", "psmask", "psrose", "psscale", "pstext", "pswiggle",
             "psxy", "psxyz"]

class GMT(object):
    def __init__(self, out, gmt5=True, **common_flags):
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
        else:
            self.common = None

    def __del__(self):
        commands = self.commands
        
        # append common flags
        if self.common is not None:
            commands = [(cmd[0] + " {}".format(self.common), cmd[1], cmd[2])
                        if cmd[0].split()[0] in _plotters else cmd
                        for cmd in commands]
        
        # add -K and -O flags
        if len(self.commands) > 1:
            commands[:-1] = [(cmd[0] + " -K", cmd[1], cmd[2])
                             if cmd[0].split()[0] in _plotters else cmd
                             for cmd in commands[:-1]]
            commands[1:] = [(cmd[0] + " -O", cmd[1], cmd[2])
                             if cmd[0].split()[0] in _plotters else cmd
                             for cmd in commands[1:]]
        
        # add "gmt " to every command if GMT version is >= 5
        if self.is_gmt5:
            commands = [("gmt " + cmd[0], cmd[1], cmd[2]) for cmd in commands]

        print("\n".join(cmd[0] for cmd in commands))
        
        # gather all the outputfiles and remove the ones that already exist
        outfiles = set(cmd[2] for cmd in commands
                       if cmd[2] is not None and pth.isfile(cmd[2]))
        for out in outfiles:
            os.remove(out)
        
        # execute gmt commands and write their output to the specified files
        for cmd in commands:
            execute_cmd(cmd)
        
        del self.commands
        del self.common
        del self.is_gmt5
        del self.out
    
    def __call__(self, gmt_exec, data=None, portrait=False, palette=None,
                 outfile=None, **flags):
        
        keys = flags.keys()
        
        cmd = gmt_exec
        
        if data is not None:
            # data is a path to a file
            if isinstance(data, string_types) and pth.isfile(data):
                cmd += " {}".format(data)
                data = None
            # data is a numpy array
            elif isinstance(data, np.ndarray):
                cmd += " -bi{}dw".format(data.shape[1])
                data = data.tobytes()
                
        # if we have flags
        if len(keys) > 0:
            gmt_flags = ["-{}{}".format(key, proc_flag(flags[key]))
                         for key in keys]
            cmd += " {}".format(" ".join(gmt_flags))
        
        if portrait:
            cmd += " -P"
        
        if palette:
            cmd += " -C{}".format(palette)
        
        if outfile is not None:
            self.commands.append((cmd, data, outfile))
        else:
            self.commands.append((cmd, data, self.out))
            
    def makecpt(self, outfile, **flags):
        cmd = "makecpt"
        
        keys = flags.keys()
        
        if len(keys) > 0:
            gmt_flags = ["-{}{}".format(key, proc_flag(flags[key]))
                         for key in keys]
            cmd += " {}".format(" ".join(gmt_flags))
        
        self.commands.append((cmd, None, outfile))
    
    def Set(self, parameter, value):
        self.commands.append(("set {}={}".format(parameter, value), None, None))

# parse GMT flags        
def proc_flag(flag):
    if isinstance(flag, list):
        return "/".join([str(elem) for elem in flag])
    elif isinstance(flag, string_types):
        return flag

# add -K or/and -O flags if the GMT command is a plotter "command"

def add_K(cmd):
    if cmd[0].split()[0] in _plotters:
        return (cmd[0] + " -K", cmd[1], cmd[2])
    else:
        return cmd

def add_O(cmd):
    if cmd[0].split()[0] in _plotters:
        return (cmd[0] + " -O", cmd[1], cmd[2])
    else:
        return cmd

def execute_cmd(cmd, ret_out=False):
    try:
        cmd_out = sub.check_output(split(cmd[0]), input=cmd[1],
                                   stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(cmd[0]))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    if cmd[2] is not None:
        with open(cmd[2], "ab") as f:
            f.write(cmd_out)
    
    if ret_out:
        return cmd_out
