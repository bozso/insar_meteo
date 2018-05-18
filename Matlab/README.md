# Matlab functions

## Introduction

I keep two Matlab function libraries here I use in conjunction with 
[StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/) and 
[TRAIN](https://github.com/dbekaert/TRAIN). Mostly helper functions.

The two Matlab modules and sfopen.m: 
- **traux.m**: [TRAIN](https://github.com/dbekaert/TRAIN) auxilliary functions,
- **staux.m**: [StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/)
  auxilliary functions,
- **sfopen.m**: safely open a file. Exits with `error` if the file cannot be
  opened.

## Calling functions

I dislike the way Matlab handles functions (each function has to be in separate
directory) so I have found a way around that. The main function in a .m file,
i.e. the first one, can call all the others. So each module has a main function
that can call all the others. The first argument of the main function is the
function name to be called, the other arguments should be the arguments to
the called module function.

For e.g.
```Matlab
>> a = ones(5,5);
>> staux('save_binary', a, 'a.dat', 'dtype', 'double');
```
will call the `save_binary` function from the **staux.m** module. See the documentation
and the code to figure out the input parameters. **staux.m** is relatively
well documented, documentation for **traux.m** is underway. For some
functions help is available and can be printed in the following way:
```Matlab
>> staux('boxplot_los', 'help');

function h = BOXPLOT_LOS(plot_flags, out, ...)

The plot will be saved to an image file defined by the `out` argument. No 
figure will pop up.
Plots the boxplot of LOS velocities defined by plot_flags.
Accepted plot flags are the same flags accepted by the ps_plot function,
with some extra rules.
    1) Multiple plot flags must be defined in a cell array, e.g.
    boxplot_los({'v-do', 'v-da'});
    2) If we have the atmospheric correction option ('v-da'), the
    cooresponding atmospheric correction flag must be defined like this:
    'v-da/a_e'. This denotes the DEM error and ERA-I corrected velocity
    values. Atmospheric coretcions can be calculated with TRAIN.

  Additional options to the boxplot function can be passed using varargin.
  - 'fun' : function to be applied to the velocity values; default value:
            nan (no function is applied); function should return a vector
            (in the case of a single plot flag) or a matrix
            (in the case of multiple plot flags).
  - 'boxplot_opt': varargin arguments for boxplot, given in a cell array;
                  e.g.: 'boxplot_opt', {'widths', 0.5, 'whisker', 2.0}
                  See the help of the boxplot function for additinal 
                  information. Default value: nan (no options)

  The function returns the function handle `h` to the boxplot.
```


## Acknowledgement

Code from [StaMPS](https://homepages.see.leeds.ac.uk/~earahoo/stamps/), [TRAIN](https://github.com/dbekaert/TRAIN) were used in the developement of
these libraries. I wish to thank Andrew Hooper and David Bekaert for
providing free and open acces to their code.
