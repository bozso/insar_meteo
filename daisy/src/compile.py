from os.path import join as pjoin
import sys

sys.path.append(pjoin("..", "..", "compile_all.py"))

from compile_all import compile_c

CC = "gcc"
flags = "-std=c99 -lm"
c_file = "daisy.c"
#depend = pjoin("..", "..", "aux", "aux_fun.c")

def main():
    compile_c(c_file, CC=CC, flags=flags)

if __name__ == "__main__":
    main()
