//#include <stdatomic.h>

#include <stdlib.h>
#include <stdio.h>


#include "ref.h"
#include "common.h"

void
_incref(const ref *refc)
{
    //atomic_fetch_add((int *)&(refc->count), 1);
    ((ref *)refc)->count++;
}

void
_decref(const ref *refc)
{
    //printf("Refcount: %d.\n", ((ref *)refc)->count);
    
    //atomic_fetch_sub((int *)&(refc->count), 1);
    if (--(((ref *)refc)->count) == 0) {
        printf("Calling destructor!\n");
        refc->free(refc->obj);
        free((void *)refc);
    }
}
