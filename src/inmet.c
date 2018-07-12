#include <stdio.h>
#include <string.h>
#include "main_functions.h"

#define Modules "azi_inc fit_orbit eval_orbit"

int main(int argc, char **argv)
{
    test_matrix();
    return 0;
    
    if (argc < 2) {
        errorln("At least one argument (module name) is required.\
                 \nModules to choose from: %s.", Modules);
        printf("Use --help or -h as the first argument to print the help message.\n");
        return err_arg;
    }
    
    if (module_select("azi_inc") || module_select("AZI_INC"))
        return azi_inc(argc, argv);
    
    else if (module_select("fit_orbit") || module_select("FIT_ORBIT"))
        return fit_orbit(argc, argv);

    else if (module_select("eval_orbit") || module_select("EVAL_ORBIT"))
        return eval_orbit(argc, argv);

    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return err_arg;
    }
    
    return 0;
}
