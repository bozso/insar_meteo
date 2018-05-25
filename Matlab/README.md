# Matlab functions

## Introduction

I keep Matlab function libraries here I use in conjunction with 
[StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/) and 
[TRAIN](https://github.com/dbekaert/TRAIN). Mostly helper functions.

The Matlab modules: 
- **Meteo.m**: [TRAIN](https://github.com/dbekaert/TRAIN) auxilliary functions,
- **Staux.m**: [StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/)
  auxilliary functions,
- **Daisy.m**: wrapper functions for calling DAISY modules, similar to daisy.py
- **gmt.m**: thin wrapper for [Generic Mapping Tools](http://gmt.soest.hawaii.edu/)

## Calling functions

I dislike the way Matlab handles functions (each function has to be in separate
directory) so I have found a way around that. I define a class in each file,
and that class conatins callable Static methods.

For e.g.
```Matlab
>> a = ones(5,5);
>> Staux.save_binary(a, 'a.dat', 'dtype', 'double');
```
will call the `save_binary` function from the **Staux.m** module. See the documentation
and the code to figure out the input parameters. **Staux.m** is relatively
well documented, documentation for **Meteo.m** is underway. For some
functions help is available and can be printed in the following way:
```Matlab
>> help Staux.boxplot_los

function h = boxplot_los(plot_flags, 'out', '', 'boxplot_opt', nan, 'fun', nan)

Plots the boxplot of LOS velocities defined by plot_flags.
accepted plot flags are the same flags accepted by the ps_plot function,
with some extra rules.
   1) Multiple plot flags must be defined in a cell array, e.g.
   boxplot_los({'v-do', 'v-da'});

   2) If we have the atmospheric correction option (''v-da''), the
   cooresponding atmospheric correction flag must be defined like this:
   'v-da/a_e'. This denotes the DEM error and ERA-I corrected velocity
   values. Atmospheric coretcions can be calculated with TRAIN.

The plot will be saved to an image file defined by the 'out' argument. No
figure will pop up, if the 'out' parameter is defined and is not an empty 
string.
 
 Additional options to the boxplot function can be passed using varargin:

'fun' : function to be applied to the velocity values; default value:
        nan (no function is applied); function should return a vector
        (in the case of a single plot flag) or a matrix
        (in the case of multiple plot flags).

'boxplot_opt': varargin arguments for boxplot, given in a cell array;
                e.g.: 'boxplot_opt', {'widths', 0.5, 'whisker', 2.0}
                See the help of the boxplot function for additinal 
                information. Default value: nan (no options)

The function returns the function handle `h` to the boxplot.
```

## Acknowledgement

Code from [StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/), 
[TRAIN](https://github.com/dbekaert/TRAIN) were used in the developement of
these libraries. I wish to thank Andrew Hooper and David Bekaert for
providing open acces to their code.
