#!/usr/bin/env python3

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

import argparse as ap
import os
import os.path as pth
import errno
import shutil as sh
import glob as gl
from contextlib import contextmanager

import inmet.cwrap as cw

_steps = frozenset(("import", "make_slcs"))

# To be translated into python

_doris__doc__=\
"""
./insar_prepoc1.sh "data_path" "processing_path" "master_date" "envi/ers/..."

At the end the script shows the radar signal intensity of the focused 
master date image. To crop the image you have to select an area that is 
present on all images. Usually areas at the edges of the master image 
should be avoided. In the case of an ascending orbit the image needs to be 
flipped horizontally. In the case of a descending orbit a vertical flip is 
needed.

asc -- flip
desc | flip

After that you can select the area based on the image. Select the first and 
last line and the first and last pixel. (Line corresponds to a row of 
pixels, pixel corresponds to a coloumn of pixels). WARNING: the image is 
multilooked! This means that if we have 20 looks in the azimuth range and 
in the image viewer the image has 1000 pixels in the azimuth, the SLC image 
actually has 20000 pixels in the azimuth. 20 looks in the azimuth and 4 
looks in range are the default values for the StaMPS ERS processing. You 
need to put into master_crop.in the non-multilooked values. The ones that 
correspond to the SLC image.

eog image.slc_4l.ras

edit master_crop.in

step_master_setup

eog "masterdate"_crop.ras

Did the program crop the image well?
Yes -- Continue. If you want to exclude some dates, edit them out from 
       make_slcs.list and after continue with insar_preproc2. 
No  -- Check master_crop.in. Did you flip the image the right way?

cd into SLC directory

make_slcs_envi/ers/...

step_master_orbit_ODR

insar_preproc2 "processing_path" "master_date" "DEM_header_path" "envi/ers/..."

The script will show the cropped image and the simulated image. If you 
judge them to be similar, continue.

Did the script print one of the date of a slave images?
Yes -- delete that SLC or check ODR path in the DORIS input file in the 
slave directory.
No  -- nothing to be done.
Check make_coarse.output!

./insar_prepoc4.sh "processing_path" "master_date"

At this point you may want to run he command master_select. It will 
calculate stack coherence and it will sort the master date candidates in 
order of their coherence. The one on the top should be th ideal master 
date. After this you may restart the preprocession with the best master date.

Check whether the script printed anything.
Yes -- Delete that slc. After: update_core in the INSAR_"masterdate"/coreg 
       directory
No  -- nothing to be done

./insar_prepoc5.sh "processing_path" "master_date"

The script will show the interferograms with tha radar amplitudes. If you 
judge them to be adquate, continue.

After this: mt_prep
"""

# $1: data path
# $2: processing path
# $3: master date
# $4: envi/ers/...
# $5: DEM header path

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise        

def data_import(params):
    
    data_path = params.get("data_path")
    proc_path = params.get("proc_path", ".")
    master_date = params.get("master_date")
    sat_type = params.get("sat_type")
    
    if data_path is None:
        raise ValueError("data_path is undefined!")
    if master_date is None:
        raise ValueError("master_date is undefined!")
    if sat_type is None:
        raise ValueError("sattelite_type is undefined!")
    
    create_dir(proc_path)
    
    print("Creating symbolic links.")
    cw.cmd("link_raw", data_path, proc_path)
    
    print("Running step_slc in master date directory.")
    with cd(pth.join(proc_path, "SLC", master_date)):
        cw.cmd("step_slc_{}".format(sat_type))
    
    my_scr = pth.expandvars("$MY_SCR")
    sh.copy(pth.join(my_scr, "master_crop.in"), ".")
    
    print("Configure master_crop.in file then continue.")
    
#               $2          $3          $4         $5
def make_slcs(proc_path, master_date, sat_type, dem_header):
    
    with cd(pth.join(proc_path, "SLC", master_date)):
        cw.cmd("step_master_setup")
    
    with cd(pth.join(proc_path, "SLC")):
        cw.cmd("make_slcs_{}".format(sat_type))
    
    with cd(proc_path):
        create_dir("DEM")
        dem_header_path="{}.hdr".format(dem_header)
    
    with cd(pth.join(proc_path, "DEM")):
        for elem in gl.glob(pth.join(dem_header + "*")):
            sh.copy(elem, ".")
    
    # select the largest one
    dem_path = sorted(gl.glob(pth.join(proc_path, "DEM")),
                      key=os.path.getsize)[0]
    dem_path = pth.join(proc_path, "DEM", dem_path)
    
    with open(dem_header, "r") as f:
        lines = f.readlines()
    
    with cd(pth.join(proc_path, "INSAR_{}".format(master_date))):
        doris_scr = pth.expandvars("$DORIS_SCR")
        sh.copy(pth.join(doris_scr, "timing.dorisin"), ".")

        return
        
        with open("timing.dorisin", "r") as f:
            timing = f.readlines()
        
        with open("timing.dorisin", "w") as f:
            f.write("\n".join(timing))
            f.write("\n")
            
#awk 'NR<29' timing.dorisin > temp
#printf "${array[0]}" >> temp
#printf "${var1[0]}\t\t $dem_path\n" >> temp
#printf "${array[2]}" >> temp
#printf "${array[3]}" >> temp
#printf "${array[4]}" >> temp
#printf "${array[5]}" >> temp
#awk 'NR>34' timing.dorisin >> temp
#sed 's/\r$//' temp > timing.dorisin
#rm temp

#step_master_orbit_ODR
#step_master_timing > timing.output
#echo "Do these images look similar?"
#eog master_sim_*.ras $1/SLC/$2/$2_crop.ras 

#cd $2/INSAR_$3

#make_orbits
#out=$(show_porbits | awk '$2==0 {print $0}')

#if [[ $out ]]; then
    #echo "Delete the SLC or check the ODR path in the DORIS input file in the \
        #slave directory of the following dates:\n"
    #echo "$out"
    #return 1
#else
    #echo "make_orbits done"
#fi

## did the awk script print one of the date of a slave image?
## yes -- delete that SLC or check ODR path in the DORIS input file in the 
## slave directory
## no  -- nothing to be done

#cd $2/INSAR_$3
#make_coarse > make_coarse.output

#make_coreg &
#make_dems &
#wait

#cd coreg
#out=$(ls -l CPM_Data* |awk '$5<1000 {print $0}')

#if [[ $out ]]; then
    #echo "make_dems done"
    #rm 
    #echo "Delete the SLC of the following date(s):\n"
    #echo "$out"
    #echo "After that run update_coreg in INSAR_$2/coreg!"
    #return 1
#else
    #echo "make_coreg and make_dems done"
#fi

#cd $2/INSAR_$3

#make_resample
#RET=$(ls -l */*.slc | gawk 'BEGIN {RET = 0}; NR==1 {SIZE = $5}; $5 != SIZE \
    #{RET = 1; print $0}; END {print RET}')

#if [ $RET -eq 1 ]; then
	#echo "One or more slc files, listed above, have different size."
	#return 1
#fi

#make_ifgs
#eog */*dem_*l.ras

## SELECT MASTER

## arg1: processing path
## arg2: master date
## arg3: envi/ers/...
## arg4: DEM header path

#cd $1/SLC
#make_slcs_$3

#cd $1/INSAR_$2
#step_master_orbit_ODR

#cp $DORIS_SCR/timing.dorisin .
#awk 'NR<29' timing.dorisin > temp
#cat $4 >> temp
#awk 'NR>34' timing.dorisin >> temp
#sed 's/\r$//' temp > timing.dorisin
#rm temp

#step_master_timing > timing.output
#make_orbits

#out=$(show_porbits | awk '$2==0 {print $0}')
#if [[ $out ]]; then
    #echo "Delete the SLC or check the ODR path in the DORIS input file in the \
        #slave directory of the following dates:\n"
    #echo "$out"
    #return 1
#else
    #echo "make_orbits done"
#fi

#master_select > master_select.out
#grep Bperp */coreg.out > bperp.out

def parse_arguments():
    parser = ap.ArgumentParser(description=_doris__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter,
            parents=[cw.gen_step_parser(_steps)])
    
    parser.add_argument(
        "-c", "--conf",
        nargs="?",
        default="doris.conf",
        type=str,
        help="Config file with processing parameters.")
    
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    start, stop = cw.parse_steps(args, _steps)
    params = cw.parse_config_file(args.conf)
    
    if start == 0:
        data_import(params)
    
    if start <= 1 and stop >= 1:
        make_slcs(params)

if __name__ == "__main__":
    main()
