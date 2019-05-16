# InSAR and Meteorology

## Introduction

Repository for storing programs I use for interferometric processing of SAR
images. I mainly use InSAR for deformation monitoring with the help of
corner reflectors and I will try to develop a method for integrated water
vapour monitoring. At the moment this is still in its infancy.

The repository now mainly holds programs written in Python and C++
(compiles into shared libraries called in Python with the `ctypes` library) that
carry out performance heavy tasks.

## Short description of folders and files

- **Matlab**: auxilliary Matlab modules for
  [StaMPS](https://github.com/dbekaert/StaMPS) and 
  [TRAIN](https://github.com/dbekaert/TRAIN),
- **inmet**: Main python scripts that call Gnuplot and my C++ modules,
- **daisy_test_data**: test datafiles for the DAISY program,
- **src**: source files for C++ shared libraries,
- **insar_meteo.sh** source this file in your .bashrc so you can use the C,
  Python and Matlab programs. **IMPORTANT**: first set the MAIN_DIR variable
  in the file.

## License

Code is licensed with [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).

## Acknowledgement

Code from [StaMPS](https://github.com/dbekaert/StaMPS),
[TRAIN](https://github.com/dbekaert/TRAIN) were used in the developement of
these libraries. I wish to thank Andrew Hooper and David Bekaert for
providing open acces to their code.
