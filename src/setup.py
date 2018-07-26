# Copyright (C) 2018  MTA CSFK GGI
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

from numpy.distutils.core import Extension, setup
from os.path import join
from shutil import move
from glob import iglob

comp_args = ["-std=c99", "-O3"]
macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

ext_insar = Extension(name="insar_aux", sources=["insar_auxmodule.c"],
                      define_macros=macros, extra_compile_args=comp_args)

setup(ext_modules=[ext_insar])

for so in iglob("*.so"):
    move(so, join("..", "inmet"))
