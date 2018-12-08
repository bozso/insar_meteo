#!/usr/bin/env python

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

_steps = ("data_select", "dominant", "poly_orbit", "integrate")

daisy__doc__  = \
"""
Commands available: {}

Usage:
    daisy.py <command> [<args>...]

Options:
    -h --help      Show this screen.
""".format(", ".join(_steps))

from inmet.utils import cmd


def data_select(argv):
    """
    Usage:
        daisy.py data_select <asc_data> <dsc_data> [--ps_sep=<sep>]
    
    Options:
        -h --help        Show this screen.
        --ps_sep=<sep>   Separation between PSs in meters [default: 100].
    """
    
    args = docopt(data_select.__doc__, argv=argv)
    print(args)

def main():
    
    cmd("/home/istvan/progs/insar_meteo/bin/fit_orbit.py")

    
if __name__ == "__main__":
    main()
