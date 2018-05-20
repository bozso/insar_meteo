#!/usr/bin/env python3

import os
from aux.gmt import GMT, plot_scatter

#infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_wet.dat"
#infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_wet.dat"
infile1 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/dinv_total.dat"
infile2 = "/mnt/bozso_i/dszekcso_ml/SMALL_BASELINES/d_total.dat"

def main():
    
    all_cols = ",".join(str(ii) for ii in range(2,73))

    #histo(infile1, "dinv_total_histo.ps", bi="72d", i=all_cols)
    #histo(infile2, "d_total_histo.ps", bi="72d", i=all_cols)
    
    # GMT configuration
    config = {"font_title":"8p", "font_annot_primary": "10p"}

    # colorbar configuration
    cbar_config = { "mode" : "h", "label": "10:Delay:/:cm:", "offset": 50}
    
    # axis_configuration
    axis_conf = {"z_range": (180.0, 280.0)}
    
    plot_scatter(infile1 , 70, "dinv_total.ps", config=config,
                 cbar_conf=cbar_conf, axis_conf=axis_conf,idx=[0, 1, 2],
                 tryaxis=True)
    
    return
    
    plot_scatter(infile2 , 70, "d_total.ps", config=config,
                 cbar_conf=cbar_conf, axis_conf=axis_conf,idx=[0, 1, 2])
        
    return 0
    
if __name__ == "__main__":
    main()
