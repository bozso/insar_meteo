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
from distutils.sysconfig import get_python_inc
from numpy.distutils.misc_util import get_numpy_include_dirs
from sysconfig import get_config_var
from glob import iglob


def make_join(root):
    from os.path import join
    
    def f(*paths):
        return join(root, *paths)
    
    return f


def main():
    mjoin = make_join("/home/istvan/miniconda3")
    rjoin = make_join("/home/istvan/progs/insar_meteo/src")

    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]

    flags = get_config_var('CFLAGS').split()
    flags.remove("-Wstrict-prototypes")
    flags += ["-std=c++11", "-Wall", "-Wextra", "-fPIC"]

    macros = []
    
    inc_dirs = [mjoin("include"), rjoin("aux"), rjoin("inmet")] + \
                get_numpy_include_dirs() + [get_python_inc()]
    lib_dirs = [mjoin("lib")]
    
    libs = ["stdc++"]
    
    print("*" * 80)
    print("*" * 80)
    print(inc_dirs)
    # return
    
    #sources = [rjoin("inmet_aux.cpp"), rjoin("aux/array.cpp")]
    # sources = [rjoin("inmet_aux.cpp"), rjoin("inmet.cpp"),
    #            rjoin("aux", "static_tpl_inst.cpp")]
    
    # sources = [rjoin("inmet.cpp"), rjoin("aux", "aux.cpp"),
    #            rjoin("inmet", "ellipsoid.cpp"), rjoin("inmet", "math.cpp")]
    
    sources = [rjoin("inmet.cpp")]
    # sources.extend(rjoin("impl", cfile)
    #                for cfile in iglob(rjoin("impl", "*.cpp")))
    
    # sources.extend(rjoin("aux" source)
                   # for source in iglob(rjoin("aux" "*.cpp")))
    
    comp = new_compiler()
    
    objects = comp.compile(sources, macros=macros, include_dirs=inc_dirs,
                           extra_preargs=flags)
    
    comp.link_shared_lib(objects, rjoin("inmet_aux"), extra_preargs=flags,
                         libraries=libs)
        
    return 0


if __name__ == "__main__":
    main()

