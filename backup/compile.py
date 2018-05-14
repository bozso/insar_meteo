#!/usr/bin/env julia

run(`gcc insar.c -o libinsar.so -fPIC -shared -Wall -Werror`)
