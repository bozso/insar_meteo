#!/usr/bin/env python3

import os
from aux.gmt import GMT, plot_scatter

infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_wet.dat"
infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_wet.dat"

def main():
    
    config = {"FONT_TITLE":"8p", "FONT_ANNOT_PRIMARY": "10p"}
    plot_scatter(infile1 , 70, "dinv_wv.ps", config=config,
                 cbar_B="10:APS:/:rad:", idx=[0, 1, 2], cbar_mode="h")
    
if __name__ == "__main__":
    main()
