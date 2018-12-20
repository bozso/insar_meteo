#ifndef COMMON_H
#define COMMON_H

#include <iso646.h>
#include <stdlib.h>
#include <stdio.h>

#define Log printf("File: %s :: Line: %d.\n", __FILE__, __LINE__)

#define Mem_New(type, num) (type *) malloc(sizeof(type) * num)
#define Mem_Malloc(num) malloc(num)
#define Mem_Free(ptr) free((ptr))


#if 1
typedef void (*dtor)(void *);

#define del(obj)                \
do{                             \
    if ((obj)) {                \
        (obj)->dtor_((obj));    \
        Mem_Free((obj));        \
        (obj) = NULL;           \
    }                           \
} while(0)
#endif

#endif
