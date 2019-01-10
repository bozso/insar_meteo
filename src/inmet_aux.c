#include <stdio.h>

#include "array.h"

int test(arptr arr)
{
    printf("%lu %lu\n", arr->shape[0], arr->strides[0]);
    return 0;
}
