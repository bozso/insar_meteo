#!/usr/bin/env python3

import os
from aux.gmt import GMT, plot_scatter, hist

#infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_wet.dat"
#infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_wet.dat"
infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_total.dat"
infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_total.dat"

def main():
    
    hist(infile1, "dinv_total_histo.ps", binary="16x70d")
    hist(infile2, "d_total_histo.ps", binary="16x70d")
    
    return
    
    # GMT configuration
    config = {"map_title_offset": "2.5p", "font_annot_primary": "10p"}

    # colorbar configuration
    cbar_config = { "mode" : "v", "label": "10:Delay:/:cm:", "offset": 25}
    
    # axis_configuration
    axis_config = {"z_range": (180.0, 280.0)}
    
    titles = ("20160212", "20160321", "20170304",
              "20160212", "20160321", "20170304")
    
    plot_scatter(infile1 , 70, "dinv_total.ps", config=config,
                 cbar_config=cbar_config, axis_config=axis_config,
                 idx=[0, 1, 2, 3, 4, 5], tryaxis=True, titles=titles)
    
    return
    
    plot_scatter(infile2 , 70, "d_total.ps", config=config,
                 cbar_conf=cbar_conf, axis_conf=axis_conf,idx=[0, 1, 2])
        
    return 0
    
if __name__ == "__main__":
    main()
