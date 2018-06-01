#!/usr/bin/env python3

import os
from inmet.gmt import GMT, plot_scatter, hist

#infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_wet.dat"
#infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_wet.dat"
infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/iwv.dat"
# infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_total.dat"

def main():
    
  #  hist(infile1, "dinv_total_histo.ps", binary="16x70d")
  #  hist(infile2, "d_total_histo.ps", binary="16x70d")
    
  #  return
    
    # GMT configuration
    config = {"map_title_offset": "2.5p", "font_annot_primary": "10p"}
    
    idx = (0, 1, 2, 3, 4, 5)
    titles = ("20160912", "20160924", "20160930", "20161006", "20161018",
              "20161024")
    
    plot_scatter(infile1 , 70, "iwv.png", config=config, idx=idx,
                 mode="v", offset=25, label="10:IWV:/:kgm-2:",
                 titles=titles, z_range=(0, 30))
    
    return
    
    plot_scatter(infile2 , 70, "d_total.ps", config=config,
                 cbar_conf=cbar_conf, axis_conf=axis_conf,idx=[0, 1, 2])
        
    return 0
    
if __name__ == "__main__":
    main()
