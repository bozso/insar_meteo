#!/usr/bin/env python3

import os
from aux.gmt import GMT, info, get_ranges

bindef = "72d"

infile = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/abs_avg.dat"

extra = 0.025

x_axis = "a0.5g0.25f0.25"
y_axis = "a0.25g0.25f0.25"

def main():
    gmt = GMT("test.ps", R=(10, 20, 45, 50), J="M10i")
    
    print(gmt.get_width())
    
    return
    
    #ll_range, c_range = get_ranges(data=infile, binary=bindef, xy_add=extra,
                                   #z_add=0.125)
    
    conf = {"FONT_TITLE":"8p", "FONT_ANNOT_PRIMARY": "10p"}
    
    gmt = GMT("test.ps", R=(10,20,30,40), j="M", config=conf, debug=True)
    x, y = gmt.multiplot(10, right=100, y_pad=190, top=180)
    
    gmt.makecpt("scatter.cpt", C="drywet", Z=True, T=c_range)
    
    for ii in range(10):
        input_format = "0,1,{}".format(ii + 2)
        gmt.psbasemap(X=x[ii], Y=y[ii], B="WSen+t{}".format(ii + 1),
                      Bx=x_axis, By=y_axis)
        #gmt.psxy(data=infile, i=input_format, bi=bindef, S="c0.025c",
                 #C="scatter.cpt")
    
    gmt.colorbar(mode="v", offset=100, C="scatter.cpt", B="10:APS:/:rad:")
    
    #xx, yy, length, width = gmt.scale_pos("v", offset=25)
    
    #gmt.psscale(D=(0.0, 0.0, length, width), Xf=xx, Yf=yy)

    del gmt
    
if __name__ == "__main__":
    main()
