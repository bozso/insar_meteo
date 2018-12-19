#ifndef FILE_H
#define FILE_H

#include <stdio.h>

#include "ref.h"

typedef struct _File {
    FILE *_file;
    ref *refc;
} File;


File
open(char const* path, char const* mode);


int
write(File const *file, char const* fmt, ...);

#endif
