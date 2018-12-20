#ifndef COMMON_H
#define COMMON_H

#include <iso646.h>

typedef void (*dtor)(void *);

#define Log printf("File: %s :: Line: %d.\n", __FILE__, __LINE__)

#define Mem_Malloc(type, num) (type *) malloc(sizeof(type) * num)

#define del(obj) (obj)->dtor_((obj))

#endif
