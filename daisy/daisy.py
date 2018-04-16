#!/usr/bin/env python3

import aux.insar_aux as ina
from gnuplot import Gnuplot, arr_plot

import numpy as np
import argparse
import faulthandler
import logging as log

faulthandler.enable()

def data_select(asc_file, dsc_file, max_sep=100.0, height_err=False):

    asc_tmp = np.loadtxt(asc_file, dtype=np.float32)
    dsc_tmp = np.loadtxt(dsc_file, dtype=np.float32)
    
    asc = np.empty((asc_tmp.shape[0], 4))
    dsc = np.empty((dsc_tmp.shape[0], 4))
    
    if height_err:
        asc[:,0:3] = asc_tmp[:,0:3]
        asc[:,3] = asc_tmp[:,3] + asc_tmp[:,4]

        dsc[:,0:3] = dsc_tmp[:,0:3]
        dsc[:,3] = dsc_tmp[:,3] + dsc_tmp[:,4]
    else:
        asc = asc_tmp[:,0:4]
        dsc = dsc_tmp[:,0:4]
    
    del asc_tmp
    del dsc_tmp
    
    idx, n_found = ina.asc_dsc_select(asc, dsc, max_sep)
    
    asc = asc[idx,:]
    
    print("Found {} ascending PSs.".format(n_found))
    
    idx, n_found = ina.asc_dsc_select(dsc, asc, max_sep)
    
    print("Found {} descending PSs.".format(n_found))
    
    np.save("asc_select.npy", asc)
    np.save("dsc_select.npy", dsc)

def plot_select(out="asc_dsc_selected.png", point_size=1.0):
    
    asc = np.load("asc_select.npy")
    dsc = np.load("dsc_select.npy")
    
    g = Gnuplot(out=out, term="pngcairo font 'Verdena,9'")
    
    g.multiplot((1,2), title="Selected ascending and descending PSs")
    
    g.title("Ascending PSs")
    g.labels(x="Longitude [deg]", y="Latitude [deg]")
    g("plot '-' {} u 1:2 with points pt 7 ps {} notitle"
      .format(arr_bin(asc), point_size))
    g(asc)
    
    g.title("Descending PSs")
    g.labels(x="Longitude [deg]", y="Latitude [deg]")
    g("plot '-' {} u 1:2 with points pt 7 ps {} notitle"
      .format(arr_bin(dsc), point_size))
    g(dsc)
    
    del g
    
def parse_args():

    parser = argparse.ArgumentParser(description="Descending Ascending "
                                     "Integrated DAISY")

    parser.add_argument("in_asc", help="text file that contains the "
                        "ASCENDING PS velocities")
    parser.add_argument("in_dsc", help="text file that contains the "
                        "DESCENDING PS velocities")


    parser.add_argument("--out_asc", help="text file that will contain the "
                        "selected ASCENDING PS velocities", nargs='?',
                        type=str, default='asc_select.xy')

    parser.add_argument("--out_dsc", help="text file that will contain the "
                        "selected DESCENDING PS velocities", nargs='?',
                        type=str, default='dsc_select.xy')


    parser.add_argument("--ps_sep", help="maximum separation distance "
                        "between ASC and DSC PS points in meters ",
                        nargs="?", type=float,
                        default=100.0)

    parser.add_argument("--logile", help="logfile name " nargs="?",
                        type=str, default="daisy.log")
    parser.add_argument("--loglevel", help="level of logging " nargs="?",
                        type=str, default="DEBUG")

    return parser.parse_args()

def main():

    # args = parse_args()
    # numeric_level = getattr(log, args.loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % loglevel)
    # log.basicConfig(filename=args.logfile, level=args.log,
    #                  level=numeric_level)

    asc = np.load("asc_select.npy")
    dsc = np.load("dsc_select.npy")
    
    print(arr_plot(asc, using="1:2", axes="x1y1")[1])
    
    #g = Gnuplot()
    #g.plot(plotter(asc[:10,:2]), plotter(dsc[:10,:2]))
    return
    # del g
    
    return
    
    # data_select("daisy/test_data/asc_data.xy", "daisy/test_data/dsc_data.xy", height_err=True)
    
    plot_select(point_size=0.5)
    
    return 0
    
if __name__ == "__main__":
    main()
