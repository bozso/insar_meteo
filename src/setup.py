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

from os.path import join
from distutils.ccompiler import new_compiler


def main():
    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]
    flags = ["-std=c11", "-O0"]
    macros = []
    libs = []
    inc_dirs = ["/home/istvan/miniconda3/include", "include", "backup"]
    lib_dirs = ["/home/istvan/miniconda3/lib"]
    
    ref = join("aux", "ref.c")
    File = join("aux", "File.c")
    sources = (ref, File, "inmet.c")
    
    comp = new_compiler()
    
    sources = comp.compile(sources, extra_preargs=flags, include_dirs=inc_dirs)
    
    comp.link_executable(sources, join("..", "bin", "inmet"), libraries=libs,
                         library_dirs=lib_dirs, extra_postargs=flags)
    

if __name__ == "__main__":
    main()


