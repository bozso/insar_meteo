#!/usr/bin/env python3

# To be translated into python

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

# arg1: data path
# arg2: processing path
# arg3: master date
# arg4: envi/ers/...
# arg5: DEM header path

if [[ $# -ne 5 ]]; then
    echo -e "* !!!"
    echo -e "*"
    echo -e "* WARNING: 5 arguments are required! Usage:\n* insar_preproc1"\
    "data_path processing_path master_date envi/ers/... DEM_file_path"
    echo -e "*"
    echo -e "* !!!\n"
    exit 1
fi

# if workdir does not exists
if [ ! -d $2 ]; then
    mkdir $2
fi

cd $2
link_raw $1 $2
cd $2/SLC/$3

step_slc_$4

cp $MY_SCR/master_crop.in .

cd $2/SLC/$3
step_master_setup
cd $2/SLC
make_slcs_$4

cd $2

if [ ! -d DEM ]; then
    mkdir DEM
fi

dem_header_path="$5.hdr"

cd $2/DEM
cp $5* .
dem_path=$(ls -S $2/DEM | head -1)
dem_path=$2/DEM/$dem_path

readarray array < $dem_header_path
var1=(${array[1]})

cd $2/INSAR_$3
cp $DORIS_SCR/timing.dorisin .
awk 'NR<29' timing.dorisin > temp
printf "${array[0]}" >> temp
printf "${var1[0]}\t\t $dem_path\n" >> temp
printf "${array[2]}" >> temp
printf "${array[3]}" >> temp
printf "${array[4]}" >> temp
printf "${array[5]}" >> temp
awk 'NR>34' timing.dorisin >> temp
sed 's/\r$//' temp > timing.dorisin
rm temp

step_master_orbit_ODR
step_master_timing > timing.output
echo "Do these images look similar?"
eog master_sim_*.ras $1/SLC/$2/$2_crop.ras 

cd $2/INSAR_$3

make_orbits
out=$(show_porbits | awk '$2==0 {print $0}')

if [[ $out ]]; then
    echo "Delete the SLC or check the ODR path in the DORIS input file in the \
        slave directory of the following dates:\n"
    echo "$out"
    return 1
else
    echo "make_orbits done"
fi

# did the awk script print one of the date of a slave image?
# yes -- delete that SLC or check ODR path in the DORIS input file in the 
# slave directory
# no  -- nothing to be done

cd $2/INSAR_$3
make_coarse > make_coarse.output

make_coreg &
make_dems &
wait

cd coreg
out=$(ls -l CPM_Data* |awk '$5<1000 {print $0}')

if [[ $out ]]; then
    echo "make_dems done"
    rm 
    echo "Delete the SLC of the following date(s):\n"
    echo "$out"
    echo "After that run update_coreg in INSAR_$2/coreg!"
    return 1
else
    echo "make_coreg and make_dems done"
fi

cd $2/INSAR_$3

make_resample
RET=$(ls -l */*.slc | gawk 'BEGIN {RET = 0}; NR==1 {SIZE = $5}; $5 != SIZE \
    {RET = 1; print $0}; END {print RET}')

if [ $RET -eq 1 ]; then
	echo "One or more slc files, listed above, have different size."
	return 1
fi

make_ifgs
eog */*dem_*l.ras

# SELECT MASTER

# arg1: processing path
# arg2: master date
# arg3: envi/ers/...
# arg4: DEM header path

cd $1/SLC
make_slcs_$3

cd $1/INSAR_$2
step_master_orbit_ODR

cp $DORIS_SCR/timing.dorisin .
awk 'NR<29' timing.dorisin > temp
cat $4 >> temp
awk 'NR>34' timing.dorisin >> temp
sed 's/\r$//' temp > timing.dorisin
rm temp

step_master_timing > timing.output
make_orbits

out=$(show_porbits | awk '$2==0 {print $0}')
if [[ $out ]]; then
    echo "Delete the SLC or check the ODR path in the DORIS input file in the \
        slave directory of the following dates:\n"
    echo "$out"
    return 1
else
    echo "make_orbits done"
fi

master_select > master_select.out
grep Bperp */coreg.out > bperp.out
