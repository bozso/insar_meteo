#include <stdio.h>
#include "aux_macros.h"

int main(int argc, char **argv)
{
    double a[2];

    while (fread(a, 2 * sizeof(double), 1, stdin) > 0) {
        println("%lf %lf", a[0], a[1]);
    }
    
    
    
    return 0;
}
