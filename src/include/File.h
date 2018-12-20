#ifndef FILE_H
#define FILE_H

#include <stdio.h>

#include "common.h"

typedef struct _File {
    FILE *_file;
    dtor dtor_;
} File;


File *
open(char const* path, char const* mode);

int
write(File const *file, char const* fmt, ...);

#endif
