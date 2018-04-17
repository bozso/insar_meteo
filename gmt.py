# import os
import subprocess as sub

# python 2/3 compatibility
from six import string_types
from shlex import split

class GMT(object):
    def __init__(self, out, gmt5=True, **common_flags):
        self.out = out
        self.commands = []
        self.is_gmt5 = gmt5
        
        keys = common_flags.keys()
        
        if len(keys) > 0:
            tmp = ["-{}{}".format(key, proc_flag(common_flags[key]))
                   for key in keys if key not in ["portrait"]]
            self.common = " ".join(tmp)
        else:
            self.common = None
    
    def __call__(self, gmt_exec, data=None, **flags):
        
        keys = flags.keys()
        
        if self.is_gmt5:
            cmd = "gmt " + gmt_exec
        else:
            cmd = gmt_exec
        
        if data is not None:
            cmd += " {}".format(data)
        
        if len(keys) > 0:
            gmt_flags = ["-{}{}".format(key, proc_flag(flags[key]))
                         for key in keys if key not in ["portrait"]]
            cmd += " {}".format(" ".join(gmt_flags))
        
        if "portrait" in keys and flags["portrait"]:
            cmd += " -P"
        
        self.commands.append(cmd)
        
    def __del__(self):
        commands = self.commands
        
        # append common flags
        if self.common is not None:
            commands = [cmd + " {}".format(self.common) for cmd in commands]
        
        # handle -K and -O flags
        if len(self.commands) > 1:
            commands[:-1] = [cmd + " -K" for cmd in commands[:-1]]
            commands[1:] = [cmd + " -O" for cmd in commands[1:]]
        
        print("\n".join(commands))
        
        # get the text to be written to the Postscript file
        ps = [execute_cmd(cmd) for cmd in commands]
        
        with open(self.out, "w") as f:
            f.write("\n".join(ps))
        
        del self.commands
        del self.common
        del self.is_gmt5
        del self.out

def proc_flag(flag):
    
    if isinstance(flag, list):
        return "/".join([str(elem) for elem in flag])
    elif isinstance(flag, string_types):
        return flag

def execute_cmd(cmd):
    try:
        out = sub.check_output(split(cmd), stderr=sub.STDOUT)
    except sub.CalledProcessError as e:
        print("ERROR: Non zero returncode from command: '{}'".format(cmd))
        print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
        print("RETURNCODE was: {}".format(e.returncode))
    
    return out.decode()
