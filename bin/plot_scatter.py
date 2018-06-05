#!/usr/bin/env python3

"""
Plot scattered points.
"""

import os
import os.path as pth
import argparse as ap

from inmet.gmt import GMT, get_ranges

def gen_list(cast):
    return lambda x: tuple(cast(elem) for elem in x.split(","))

def parse_arguments():
    parser = ap.ArgumentParser(description=__doc__,
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("infile", help="Binary inputfile.", type=str)
    parser.add_argument("ncols", help="Number of data columns in inputfile.",
                        type=int)
    
    parser.add_argument("--out", help="Postscript or raster outputfile.",
                        default=None, nargs="?", type=str)
    parser.add_argument("-p", "--proj", help="Projection of the maps.",
                        default="J", nargs="?", type=str)
    parser.add_argument("-i", "--idx", help="Column indices to work with. "
                        "List of integers.", default=None, nargs="?",
                        type=gen_list(int))
    parser.add_argument("--titles", help="Titles of the plots. "
                        "List of strings.", default=None, nargs="?",
                        type=gen_list(str))
    
    parser.add_argument("--tryaxis", action="store_true")
    
    parser.add_argument("-r", "--right", help="Right margin in points.",
                        default=100, nargs="?", type=float)
    parser.add_argument("-l", "--left", help="Left margin in points.",
                        default=50, nargs="?", type=float)
    parser.add_argument("-t", "--top", help="Top margin in points.",
                        default=0, nargs="?", type=float)

    parser.add_argument("--xy_range", help="Range of x and y axis. "
                        "List of floats.", default=None, nargs="?",
                        type=gen_list(float))
    parser.add_argument("--z_range", help="Range of z values. "
                        "List of floats.", default=None, nargs="?",
                        type=gen_list(float))

    parser.add_argument("--cpt", help="GMT colorscale.", default="drywet",
                        nargs="?", type=str)

    parser.add_argument("--x_axis", help="GMT configuration of x axis.",
                        default="a0.5g0.25f0.25", nargs="?", type=str)
    parser.add_argument("--y_axis", help="GMT configuration of y axis.",
                        default="a0.25g0.25f0.25", nargs="?", type=str)
    
    parser.add_argument("--xy_add", help="Extension of x and y range.",
                        default=0.05, nargs="?", type=float)
    parser.add_argument("--z_add", help="Extension of z range.",
                        default=0.1, nargs="?", type=float)
    
    parser.add_argument("--mode", help="Colorbar mode, set it to v for "
                        "vertical or h for horizontal.", nargs="?", default="v",
                        type=str)
    parser.add_argument("--label", help="Colorbar label.", nargs="?",
                        default="", type=str)
    parser.add_argument("--offset", help="Colorbar offset towards the margins. ",
                        nargs="?", default=10.0, type=float)
    
    return parser.parse_args()
    
def main():
    args = parse_arguments()
    
    # print(args); return 0
    
    infile = args.infile
    
    if args.out is None:
        out = pth.basename(args.infile).split(".")[0] + ".png"
    else:
        out = args.out
    
    name, ext = pth.splitext(out)
    
    if ext != ".ps":
        ps_file = name + ".ps"
    else:
        ps_file = out
    
    # 2 additional coloumns for coordinates, float64s are expected
    bindef = "{}d".format(args.ncols + 2)
    
    if args.xy_range is None or args.z_range is None:
        _xy_range, _z_range = get_ranges(data=infile, binary=bindef,
                                         xy_add=args.xy_add, z_add=args.z_add)
    
    if args.xy_range is None:
        xy_range = _xy_range
    else:
        xy_range = args.xy_range

    if args.z_range is None:
        z_range = _z_range
    else:
        z_range = args.z_range
        
    if args.idx is None:
        idx = range(args.ncols)
    else:
        idx = args.idx
    
    if args.titles is None:
        titles = range(1, args.ncols + 1)
    else:
        titles = args.titles
    
    gmt = GMT(ps_file, R=xy_range)
    x, y = gmt.multiplot(len(idx), args.proj, right=args.right, top=args.top,
                         left=args.left)
    
    gmt.makecpt("tmp.cpt", C=args.cpt, Z=True, T=z_range)
    
    # do not plot the scatter points yet just test the placement of basemaps
    if args.tryaxis:
        for ii in idx:
            input_format = "0,1,{}".format(ii + 2)
            gmt.psbasemap(Xf="{}p".format(x[ii]), Yf="{}p".format(y[ii]),
                          B="WSen+t{}".format(titles[ii]),
                          Bx=args.x_axis, By=args.y_axis)
    else:
        for ii in idx:
            input_format = "0,1,{}".format(ii + 2)
            gmt.psbasemap(Xf="{}p".format(x[ii]), Yf="{}p".format(y[ii]),
                          B="WSen+t{}".format(titles[ii]),
                          Bx=args.x_axis, By=args.y_axis)

            gmt.psxy(data=infile, i=input_format, bi=bindef, S="c0.025c",
                     C="tmp.cpt")
    
    if args.label == "":
        gmt.colorbar(mode=args.mode, offset=args.offset, C="tmp.cpt")
    else:
        gmt.colorbar(mode=args.mode, offset=args.offset, B=args.label, C="tmp.cpt")
    
    if ext != ".ps":
        gmt.raster(out)
        os.remove(ps_file)
    
    os.remove("tmp.cpt")
    
    del gmt
    
    return 0
    
if __name__ == "__main__":
    main()
