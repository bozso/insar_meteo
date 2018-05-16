#!/usr/bin/env python3

import os
from aux.gmt import GMT, info, get_ranges

bindef = "72d"

infile = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/abs_avg.dat"

extra = 0.025

x_axis = "a0.5g0.25f0.25"
y_axis = "a0.25g0.25f0.25"

def main():
    
    ll_range, c_range = get_ranges(data=infile, binary=bindef, xy_add=extra,
                                   z_add=0.125)
    
    
    gmt = GMT("test.ps", R=ll_range)
    x, y = gmt.multiplot(8, "M", y_pad=180, top=200)
    
    gmt.makecpt("scatter.cpt", C="drywet", Z=True, T=c_range)
    
    for ii in range(8):
        input_format = "0,1,{}".format(ii + 2)
        gmt.psbasemap(X=x[ii], Y=y[ii], B="WSen", Bx=x_axis, By=y_axis)
        gmt.psxy(data=infile, i=input_format, bi=bindef, S="c0.025c",
                 C="scatter.cpt")
    
    gmt.psscale(D=(22, 9.5, 15, 0.7), C="scatter.cpt", B="10:APS:/:rad:")

    del gmt
    
if __name__ == "__main__":
    main()
