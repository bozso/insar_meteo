#!/usr/bin/env python3

import aux.insar_aux as ina

import numpy as np
import argparse

def data_select(

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

    
if __name__ == "__main__":
    main()
