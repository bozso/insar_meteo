INCLUDE=''
DEFINE=""
CFLAGS="-fexceptions -fno-omit-frame-pointer -pthread -shared -fPIC -std=c99 -O3"
LIBS="-lm"
LIBPATH=''
gcc insar.c $LIBPATH $LIBS $CFLAGS $DEFINE $INCLUDE -o libinsar.so 
#mex -v COMPFLAGS='$COMPFLAGS -Wall -std=c99' insar.c
# 
