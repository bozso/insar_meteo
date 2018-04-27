#!/usr/bin/env python3

import os
from gmt import GMT, info, get_ranges

bindef = "72d"

infile = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/abs_avg.dat"

extra = 0.025

axis = "a0.25g0.25f0.125"

def main():
    
    ll_range, c_range = get_ranges(data=infile, binary=bindef,
                                   xy_add=extra, z_add=0.125)
    
    gmt = GMT("abs_avg.ps", R=ll_range, J="M7.5i")
    
    print(gmt.get_config("PS_MEDIA"))
    
    del gmt; return
    
    gmt.makecpt("scatter.cpt", C="drywet", Z=True, T=c_range)
    gmt.psbasemap(B="WSen+tAAA", Bx=axis, By=axis)
    gmt.psxy(data=infile, i="0,1", bi=bindef, S="c0.025c", C="scatter.cpt")
    gmt.psscale(D=(22, 9.5, 15, 0.7), C="scatter.cpt", B="10:APS:/:rad:")

    del gmt
    
if __name__ == "__main__":
    main()
