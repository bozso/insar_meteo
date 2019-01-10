#include <stdio.h>


#include "array.h"
#include "view.h"

extern "C" {

int test(arptr _arr)
{
    view<int64_t> const arr{_arr};
    
    printf("%lu\n", arr.ndim);
    
    for(size_t ii = 0; ii < arr.shape[0]; ++ii)
    {
        for(size_t jj = 0; jj < arr.shape[1]; ++jj)
            printf("%ld ", arr(ii, jj));
        printf("\n");
    }
    
    printf("\n");

    return 0;
}

}
