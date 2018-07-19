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

from os.path import join as pjoin
from inmet.cwrap import cmd
from distutils.ccompiler import new_compiler

f_file = "inmet.f95"
libs = [":libgfortran.so.3"]
lib_dirs = ["/home/istvan/miniconda3/lib"]
#flags = ["-std=c99", "-Ofast", "-march=native", "-ffast-math"]
flags = None

def main():
    
    sources = ["utils.f95", "main_functions.f95", f_file]
    
    f_basename = f_file.split(".")[0]
    obj = []
    
    comp = new_compiler()
    exe_name = comp.executable_filename(f_basename)
    
    # trick the compiler into thinking fortran files are c files
    comp.src_extensions.append(".f95")
    
    obj = [comp.compile([source])[0] for source in sources]
    
    comp.link_executable(obj,
                         pjoin("..", "bin", exe_name),
                         libraries=libs, library_dirs=lib_dirs,
                         extra_postargs=flags)

if __name__ == "__main__":
    main()

