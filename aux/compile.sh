INCLUDE=''
DEFINE=""
CFLAGS="-fexceptions -fno-omit-frame-pointer -pthread -shared -fPIC -std=c99 -O3"
LIBS="-lm"
LIBPATH=''
BASE=$(basename $1 | cut -d'.' -f1)
gcc insar.c $LIBPATH $LIBS $CFLAGS $DEFINE $INCLUDE -o lib$BASE.so 
