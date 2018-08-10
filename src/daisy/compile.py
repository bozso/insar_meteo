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
from glob import iglob
from os import remove
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from inmet.compilers import compile_project

def parse_args():
    
    ap = ArgumentParser(description=__doc__, formatter_class=
                        ArgumentDefaultsHelpFormatter)
    
    ap.add_argument(
        "--clean",
        action="store_true",
        help="If defined the program will clean the object files after "
             "compilation.")
    
    return ap.parse_args()

def main():

    args = parse_args()
    
    flags = ["-O3", "-march=native"]
    
    compile_project("daisy.c", outdir=join("..", "..", "bin"), libs=["m"],
                    flags=flags)

    if args.clean:
        print("\nCleaning up object file.", end="\n\n")
        remove("daisy.o")


if __name__ == "__main__":
    main()
