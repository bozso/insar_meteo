#include <stdio.h>
#include "aux_macros.h"

int main(int argc, char **argv)
{
    double a[3];

    while (fread(a, 3 * sizeof(double), 1, stdin) > 0) {
        println("%lf %lf %lf", a[0], a[1], a[2]);
    }
    
    
    
    return 0;
}
