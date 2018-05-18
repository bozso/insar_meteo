# InSAR and Meteorology

## Introduction

Repository for storing programs I use for interferometric processing of SAR
images. I mainly use InSAR for deformation monitoring with the help of
corner reflectors and I will try to develop a method for integrated water
vapour monitoring. At the moment this is still in its infancy.

The repository now mainly holds programs written in Python and C (both pure
C, i.e. binary callable from the command line: `daisy.c` and C Python-extension
that is callable from python: `insar_auxmodule.c`) that will form the basis
for my future programs.

## Short description of folders and files

- **Matlab**: auxilliary Matlab modules for
  [StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/) and 
  [TRAIN](https://github.com/dbekaert/TRAIN).
- **aux**: Auxilliary C and Python functions/modules
    - **capi_macros.h** -- useful macros for writing C Python and Numpy extensions
    - **gmt.py** -- thin wrapper for [Generic Mapping Tools](http://gmt.soest.hawaii.edu/)
    - **gmtpy.py** -- inspiration for gmt.py from emolch <https://github.com/emolch/gmtpy>
    - **insar_auxmodule.c** -- auxilliary functions callable from python
    - **satorbit.py** -- module for fitting polynoms to SAR orbit data (t,x,y,z)
    - **compile.py** -- script for compiling insar_auxmodule.c
    - **tkplot.py** -- simple 2D plots using tkinter

- **backup**: I store files here that I do not use anymore, but may be useful in the
  future
- **daisy_test_data**: test datafiles for the daisy program
- **src**: source file(s) for C programs
- **insar_meteo.sh** source this file in your .bashrc so you can use the C and
  Python programs. **IMPORTANT**: first set the MAIN_DIR variable in the file.

## License

Code is licensed with [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).
