_is_gmt5 = True

# import os
import subprocess as sub

# python 2/3 compatibility
from six import string_types
from shlex import split
from sys import exit

class GMT(object):
    def __init__(self, out, proj=None, region=None):
        self.out = out
        self.region = region
        self.proj = proj
        self.commands = []
    
    def __call__(self, gmt_exec, data=None, **flags):
        
        _keys = flags.keys()
        
        if _is_gmt5:
            cmd = "gmt " + gmt_exec
        else:
            cmd = gmt_exec
        
        if data is not None:
            cmd += " {}".format(data)
        
        gmt_flags = ["-{}{}".format(key, proc_flag(flags[key]))
                     for key in _keys if key not in ["portrait"]]
        
        cmd += " {}".format(" ".join(gmt_flags))
        
        if "portrait" in _keys and flags["portrait"]:
            cmd += " -P"
        
        self.commands.append(cmd)
        
    def __del__(self):
        out = self.out
        
        if self.proj:
            self.commands = [cmd + " -J{}".format(self.proj)
                             for cmd in self.commands]

        if self.region:
            self.commands = [cmd + " -R{}".format(proc_flag(self.region))
                             for cmd in self.commands]

        if len(self.commands) > 1:
            self.commands[:-1] = [cmd + " -K" for cmd in self.commands[:-1]]
            self.commands[1:] = [cmd + " -O" for cmd in self.commands[1:]]
        
        self.commands[0] += " > {}".format(out)
        self.commands[1:] = [cmd + " >> {}".format(out)
                             for cmd in self.commands[1:]]
        
        print("\n".join(self.commands))
        for cmd in self.commands:
            try:
                proc = sub.check_output(split(cmd), stderr=sub.STDOUT)
            except sub.CalledProcessError as e:
                print("ERROR: Non zero returncode from command: '{}'".format(cmd))
                print("OUTPUT OF THE COMMAND: \n{}".format(e.output.decode()))
                print("RETURNCODE was: {}".format(e.returncode))
                
        del self.commands
        del self.out

def proc_flag(flag):
    
    if isinstance(flag, list):
        return "/".join([str(elem) for elem in flag])
    elif isinstance(flag, string_types):
        return flag

"""
def GMT(util, data=None, out=None, **gmt_flags):
   """  """
   ret = None
   
   # build command
   if _is_gmt5:
       cmd = "gmt " + os.path.join(GMT_BIN, util)
   else:
       cmd = os.path.join(GMT_BIN, util)
       
   if isinstance(data, string_types):
      cmd += ' ' + data
   for key in gmt_flags:
      flag = '-' + key
      if gmt_flags[key] not in ['', None]:
         if hasattr(gmt_flags[key], '__iter__'):
            gmt_flags[key] = '/'.join([str(item) for item in gmt_flags[key]])
         flag += gmt_flags[key]
      cmd += ' ' + flag
   if out not in ['stdout', None]:
      d = ' > '
      if '-O' in gmt_flags.keys():
         d = ' >> '
      cmd += d+out
      
   # prepare data
   d = None
   if hasattr(data,'__iter__'):
      d = []
      for item in data:
         if hasattr(item,'__iter__'):
            d.append(' '.join([str(jtem) for jtem in item]))
         else:
            d.append(str(item))
      d = '\n'.join(d)
      
   print(cmd)
   # set up job
   if out == None:
      if d is not None:
         pid = Popen(cmd,stdout=PIPE,stdin=PIPE,shell=True)
      else:
         pid = Popen(cmd,stdout=PIPE,shell=True)
   else:
      if d is not None:
         pid = Popen(cmd,stdin=PIPE,shell=True)
      else:
         pid = Popen(cmd,shell=True)
   if d is not None:
      # write data to stdin
      pid.stdin.write(d)
      
   # run it
   stdout = pid.communicate()
   if out is None:
      ret = stdout[0]
   return ret

def blockmean(*args,**kwargs):     return GMT("blockmean",*args,**kwargs)
def blockmedian(*args,**kwargs):   return GMT("blockmedian",*args,**kwargs)
def blockmode(*args,**kwargs):     return GMT("blockmode",*args,**kwargs)
def filter1d(*args,**kwargs):      return GMT("filter1d",*args,**kwargs)
def grdfilter(*args,**kwargs):     return GMT("grdfilter",*args,**kwargs)
def fitcircle(*args,**kwargs):     return GMT("fitcircle",*args,**kwargs)
def gmt2rgb(*args,**kwargs):       return GMT("gmt2rgb",*args,**kwargs)
def gmtconvert(*args,**kwargs):    return GMT("gmtconvert",*args,**kwargs)
def gmtlogo(*args,**kwargs):       return GMT("gmtlogo",*args,**kwargs)
def gmtmath(*args,**kwargs):       return GMT("gmtmath",*args,**kwargs)
def gmtselect(*args,**kwargs):     return GMT("gmtselect",*args,**kwargs)
def grd2cpt(*args,**kwargs):       return GMT("grd2cpt",*args,**kwargs)
def grd2xyz(*args,**kwargs):       return GMT("grd2xyz",*args,**kwargs)
def grdblend(*args,**kwargs):      return GMT("grdblend",*args,**kwargs)
def grdclip(*args,**kwargs):       return GMT("grdclip",*args,**kwargs)
def grdcontour(*args,**kwargs):    return GMT("grdcontour",*args,**kwargs)
def grdcut(*args,**kwargs):        return GMT("grdcut",*args,**kwargs)
def grdedit(*args,**kwargs):       return GMT("grdedit",*args,**kwargs)
def grdfft(*args,**kwargs):        return GMT("grdfft",*args,**kwargs)
def grdgradient(*args,**kwargs):   return GMT("grdgradient",*args,**kwargs)
def grdhisteq(*args,**kwargs):     return GMT("grdhisteq",*args,**kwargs)
def grdimage(*args,**kwargs):      return GMT("grdimage",*args,**kwargs)
def grdinfo(*args,**kwargs):       return GMT("grdinfo",*args,**kwargs)
def grdlandmask(*args,**kwargs):   return GMT("grdlandmask",*args,**kwargs)
def grdmask(*args,**kwargs):       return GMT("grdmask",*args,**kwargs)
def grdmath(*args,**kwargs):       return GMT("grdmath",*args,**kwargs)
def grdpaste(*args,**kwargs):      return GMT("grdpaste",*args,**kwargs)
def grdproject(*args,**kwargs):    return GMT("grdproject",*args,**kwargs)
def grdreformat(*args,**kwargs):   return GMT("grdreformat",*args,**kwargs)
def grdsample(*args,**kwargs):     return GMT("grdsample",*args,**kwargs)
def grdtrack(*args,**kwargs):      return GMT("grdtrack",*args,**kwargs)
def grdtrend(*args,**kwargs):      return GMT("grdtrend",*args,**kwargs)
def grdvector(*args,**kwargs):     return GMT("grdvector",*args,**kwargs)
def grdview(*args,**kwargs):       return GMT("grdview",*args,**kwargs)
def grdvolume(*args,**kwargs):     return GMT("grdvolume",*args,**kwargs)
def greenspline(*args,**kwargs):   return GMT("greenspline",*args,**kwargs)
def makecpt(*args,**kwargs):       return GMT("makecpt",*args,**kwargs)
def mapproject(*args,**kwargs):    return GMT("mapproject",*args,**kwargs)
def minmax(*args,**kwargs):        return GMT("minmax",*args,**kwargs)
def nearneighbor(*args,**kwargs):  return GMT("nearneighbor",*args,**kwargs)
def project(*args,**kwargs):       return GMT("project",*args,**kwargs)
def ps2raster(*args,**kwargs):     return GMT("ps2raster",*args,**kwargs)
def psbasemap(*args,**kwargs):     return GMT("psbasemap",*args,**kwargs)
def psbbox(*args,**kwargs):        return GMT("psbbox",*args,**kwargs)
def psclip(*args,**kwargs):        return GMT("psclip",*args,**kwargs)
def pscoast(*args,**kwargs):       return GMT("pscoast",*args,**kwargs)
def pscontour(*args,**kwargs):     return GMT("pscontour",*args,**kwargs)
def pshistogram(*args,**kwargs):   return GMT("pshistogram",*args,**kwargs)
def psimage(*args,**kwargs):       return GMT("psimage",*args,**kwargs)
def pslegend(*args,**kwargs):      return GMT("pslegend",*args,**kwargs)
def psmask(*args,**kwargs):        return GMT("psmask",*args,**kwargs)
def psrose(*args,**kwargs):        return GMT("psrose",*args,**kwargs)
def psscale(*args,**kwargs):       return GMT("psscale",*args,**kwargs)
def pstext(*args,**kwargs):        return GMT("pstext",*args,**kwargs)
def pswiggle(*args,**kwargs):      return GMT("pswiggle",*args,**kwargs)
def psxy(*args,**kwargs):          return GMT("psxy",*args,**kwargs)
def psxyz(*args,**kwargs):         return GMT("psxyz",*args,**kwargs)
def sample1d(*args,**kwargs):      return GMT("sample1d",*args,**kwargs)
def spectrum1d(*args,**kwargs):    return GMT("spectrum1d",*args,**kwargs)
def splitxyz(*args,**kwargs):      return GMT("splitxyz",*args,**kwargs)
def surface(*args,**kwargs):       return GMT("surface",*args,**kwargs)
def trend1d(*args,**kwargs):       return GMT("trend1d",*args,**kwargs)
def trend2d(*args,**kwargs):       return GMT("trend2d",*args,**kwargs)
def triangulate(*args,**kwargs):   return GMT("triangulate",*args,**kwargs)
def xyz2grd(*args,**kwargs):       return GMT("xyz2grd",*args,**kwargs)


def config(gmt_bin,save=False):
   function config() configures the path to the GMT executables
   
   INPUT:
    gmt_bin     path to GMT bin directory, if an empty string
                  users standard paths will be searched
    save        if true function will attempt to save path
                  by rewritting module
                  
   RETURN:
    function return:
     0    if all went well
     1    if could not read module src
     2    if could not update module src
     3    if could not write updated module

   global GMT_BIN
   GMT_BIN = gmt_bin
   ret = 0
   if save:
      try:
         mod = open(__file__,'r').readlines()
      except:
         ret = 1
      else:
         upd = False
         for i in range(len(mod)):
            if mod[i][:7] == "GMT_BIN" and mod[i][7:].strip()[0] == '=':
               mod[i] = 'GMT_BIN="%s"\n' % (gmt_bin)
               upd = True
               break
         if not upd:
            ret = 2
         else:
            try:
               open(__file__,'w').write(''.join(mod))
            except:
               ret=3
   return ret
"""
