//#include <stdatomic.h>

#include "ref.h"
#include <stdlib.h>

void
_incref(const ref *refc)
{
    //atomic_fetch_add((int *)&(refc->count), 1);
    ((ref *)refc)->count++;
}

void
_decref(const ref *refc)
{
    if (--(((ref *)refc)->count) == 0) {
        refc->free(refc->obj);
        free((void *)refc);
    }
}
