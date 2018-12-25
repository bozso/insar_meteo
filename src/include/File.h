#ifndef FILE_H
#define FILE_H

#include <stdio.h>

#include "common.h"

extern_begin

struct _File {
    FILE *_file;
    dtor dtor_;
};

typedef struct _File* File;

File
open(char const* path, char const* mode);

int
write(File const file, char const* fmt, ...);

int
read(File const file, char const* fmt, ...);

size_t
writeb(File const file, size_t const size, size_t const count, void const* data)

size_t
readb(File const file, size_t const size, size_t const count, void const* data)


extern_end

#endif
