#!/usr/bin/env julia

run(`gcc aux.c -o libaux.so -fPIC -shared -Wall -Werror`)
