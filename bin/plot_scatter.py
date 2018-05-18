#!/usr/bin/env python3

import os
from aux.gmt import GMT, plot_scatter

#infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_wet.dat"
#infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_wet.dat"
infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_total.dat"
infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_total.dat"

def main():
    z_range = (180.0, 300.0)
    
    config = {"FONT_TITLE":"8p", "FONT_ANNOT_PRIMARY": "10p"}
    plot_scatter(infile1 , 70, "dinv_total.ps", config=config,
                 cbar_B="10:Delay:/:cm:", idx=[0, 1, 2], cbar_mode="h",
                 z_range=z_range)
    plot_scatter(infile2 , 70, "d_total.ps", config=config,
                 cbar_B="10:Delay:/:cm:", idx=[0, 1, 2], cbar_mode="h",
                 z_range=z_range)
    
    return
    
    plot_scatter(infile1 , 70, "dinv_wv.ps", config=config,
                 cbar_B="5:Delay:/:cm:", idx=[0, 1, 2], cbar_mode="h",
                 z_range=z_range)
    plot_scatter(infile2 , 70, "d_wv.ps", config=config,
                 cbar_B="5:Delay:/:cm:", idx=[0, 1, 2], cbar_mode="h",
                 z_range=z_range)
    
if __name__ == "__main__":
    main()
