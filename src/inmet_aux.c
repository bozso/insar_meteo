#include <stdio.h>

#include "nparray.h"
#include "utils.h"
#include "common.h"


int main(void)
{
    nparray *arr = from_data(np_double, (double[]){1.0, 2.0}, 1, (npy_intp[]){1, 2});
    del(arr);
    
    return 0;
}
