#!/usr/bin/env python3

import daisy as dy

import numpy as np
import argparse


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

    return parser.parse_args()

def main():

    args = parse_args()

    in_asc = args.in_asc
    in_dsc = args.in_dsc
    out_asc = args.out_asc
    out_dsc = args.out_dsc

    ps_sep = args.ps_sep

    dy.data_select(in_asc, in_dsc, out_asc=out_asc, out_dsc=out_dsc,
                   max_diff=ps_sep)

    dy.dominant(out_asc, out_dsc, max_diff=ps_sep)

    dy.poly_orbit("asc_master.res", 4)
    dy.poly_orbit("dsc_master.res", 4)

    dy.integrate()

    dy.zero_select("integrate_nonzero.xy", "integrate_zero.xy")

if __name__ == "__main__":
    main()
