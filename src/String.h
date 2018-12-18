#ifndef STRING_H
#define STRING_H

#include "object.h"

typedef struct _string {
    const var class;
    char *str;
    size_t len;
} string;

extern const var String;

#endif
