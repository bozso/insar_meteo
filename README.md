# InSAR and Meteorology

Repository for storing programs I use for interferometric processing of SAR
images. I mainly use InSAR for deformation monitoring with the help of
corner reflectors and I will try to develop a method for integrated water
vapour monitoring. At the moment this is still in its infancy.

The repository now mainly holds programs written in Python and C (both pure
C, i.e. binary callable from the command line: `daisy.c` and C Python-extension
that is callable from python: `insar_auxmodule.c`) that are will form the basis
for my future programs.
