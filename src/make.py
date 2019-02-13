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

from distutils.ccompiler import new_compiler, show_compilers, get_default_compiler
from sysconfig import get_config_var

def make_join(root):
    from os.path import join
    
    def f(path):
        return join(root, path)
    
    return f


def main():
    #print(get_default_compiler(), show_compilers())
    # return
    
    mjoin = make_join("/home/istvan/miniconda3")
    rjoin = make_join("/home/istvan/progs/insar_meteo/src")

    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]

    flags = get_config_var('CFLAGS').split()
    flags.remove("-Wstrict-prototypes")
    flags += ["-std=c++11", "-Wall", "-Wextra", "-fPIC"]

    macros = []
    
    inc_dirs = [mjoin("include"), rjoin("aux")]
    lib_dirs = [mjoin("lib")]
    
    libs = ["stdc++"]
    
    if 1:
        #sources = [rjoin("inmet_aux.cpp"), rjoin("aux/array.cpp")]
        sources = [rjoin("inmet_aux.cpp"), rjoin("tpl_inst.cpp"),
                   rjoin("aux/stl_inst.cpp"), rjoin("aux/array.cpp")]
        
        comp = new_compiler()
        
        objects = comp.compile(sources, macros=macros, include_dirs=inc_dirs,
                               extra_preargs=flags)
        
        #lib = comp.library_filename("inmet_aux", lib_type="shared",
                                    #output_dir=".")
        
        comp.link_shared_lib(objects, "inmet_aux", extra_preargs=flags,
                             libraries=libs)


    if 0:
        macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        sources = ["inmet_auxmodule.c", "implement.c"]
    
        ext_modules = [
            Extension(name="inmet_aux", sources=sources,
                      define_macros=macros,
                      extra_compile_args=flags,
                      include_dirs=inc_dirs)
        ]
    
        setup(ext_modules=ext_modules)
    
    return 0


if __name__ == "__main__":
    main()


