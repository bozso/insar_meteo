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

from distutils.ccompiler import new_compiler
from os.path import basename, join

def compile_project(c_file, *adds, macros=None, flags=None, inc_dirs=None,
                    lib_dirs=None, libs=None, outdir="."):
    
    c_basename = basename(c_file).split(".")[0]

    sources = [c_file]
    sources.extend(adds)
    
    ccomp = new_compiler()
    obj = [ccomp.compile([source], extra_postargs=flags, include_dirs=inc_dirs,
                         macros=macros)[0] for source in sources]
    
    ccomp.link_executable(obj, join(outdir, c_basename), libraries=libs,
                          library_dirs=lib_dirs, extra_postargs=flags)


def compile_exe(obj, *objects, macros=None, flags=None, inc_dirs=None,
                lib_dirs=None, libs=None, outdir="."):
    
    objs = [obj]
    objs.extend(objects)
    
    ccomp = new_compiler()
    
    ccomp.link_executable(obj, join(outdir, c_basename), libraries=libs,
                          library_dirs=lib_dirs, extra_postargs=flags)


def compile_object(*sources, macros=None, flags=None, inc_dirs=None):
    
    ccomp = new_compiler()
    
    return [ccomp.compile([source], extra_postargs=flags, include_dirs=inc_dirs,
                          macros=macros)[0] for source in sources]
