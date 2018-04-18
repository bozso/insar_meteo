#include <stdio.h>

#include "aux_macros.h"

int main(int argc, char **argv)
{
    double a[3];
    //println("J0(%g) = %.18e", 5.0, gsl_sf_bessel_J0(5.0));
    
    //return 0;
    
    /*for(double ii = 0.0; ii < 1e6; ii++) {
        a[0] = 0.0;
        a[1] = ii;
        a[2] = ii * 2;
        fwrite(a, 3 * sizeof(double), 1, stdout);
    }*/
    
    while (fread(a, 3 * sizeof(double), 1, stdin) > 0) {
        //fwrite(a, 3 * sizeof(double), 1, stdout);
        println("%lf %lf %lf", a[0], a[1], a[2]);
    }
    
    
    
    return 0;
}
