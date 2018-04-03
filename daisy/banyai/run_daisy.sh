#!/bin/sh

ps_data_select $1 $2 $3
ps_dominant $1s $2s $3

ps_poly_orbit $4 $6
ps_poly_orbit $5 $6

ds_integrate dominant.xyd $(basename $4 .res).porb $(basename $5 .res).porb
