#ifndef COMMON_H
#define COMMON_H

#include <iso646.h>
#include <stdlib.h>
#include <stdio.h>

#define m_log printf("File: %s -- Line: %d.\n", __FILE__, __LINE__)

#define Mem_New(type, num) (type *) malloc(sizeof(type) * num)
#define Mem_Malloc(num) malloc(num)
#define Mem_Free(ptr) free((ptr))


#define m_check_fail(condition) \
do {                            \
    if ((condition))            \
        goto fail;              \
} while(0)


#define m_check(condition, expression) \
do {                                   \
    if ((condition))                   \
        (expression);                  \
} while(0)


#ifdef __cplusplus
#define extern_begin extern "C" {
#define extern_end }
#else
#define extern_begin
#define extern_end
#endif


typedef void (*dtor)(void *);

#if 1
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
