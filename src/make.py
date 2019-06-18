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
from numpy.distutils.core import setup
from distutils.sysconfig import get_config_var
from sysconfig import get_config_var
from glob import iglob


from distutils.core import Extension
from distutils.command.build_ext import build_ext


class build_ext(build_ext):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypes)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


class CTypes(Extension): pass


def main():
    delim = "*" * 80
    print("{:^80}\n*{:^80}*\n{:^80}".format(delim, "Compilation start.", delim))
    
    conda = "/home/istvan/miniconda3"

    #flags = ["-std=c++03", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    #flags = ["-std=c++03", "-O0", "-save-temps"]

    flags = set(get_config_var('CFLAGS').split())
    flags.remove("-Wstrict-prototypes")
    flags |= {"-std=c++11", "-Wall", "-Wextra"}
    flags = list(flags)
    
    macros = []
    inc_dirs = ["aux", "inmet"]
    # lib_dirs = [mjoin("lib")]
    libs = ["stdc++"]
    
    
    modules = [
        CTypes("inmet_aux",
               sources=["inmet.cpp"], 
               include_dirs=inc_dirs,
               extra_compile_args=flags,
               language="c++"
        )
    ]
    
    setup(ext_modules=modules, cmdclass={'build_ext': build_ext})
        
    return 0


if __name__ == "__main__":
    main()

