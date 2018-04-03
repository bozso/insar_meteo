#!/bin/sh

flags="-lm -O3 -std=c99 -Wstrict-prototypes -DCPU_LITTLE_END -mtune=native -msse2 -mfpmath=sse -Wsign-compare -g -fwrapv"

for source in $(ls *.c); do
    gcc $source -o $(basename $source .c) $flags
done
