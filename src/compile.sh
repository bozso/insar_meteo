CC=gcc
INCLUDE='../aux'
FLAGS='-O3 -std=c99'
LIBS='-lm'
BASE=$(basename $1 | cut -d'.' -f1)
$CC $1 $LIBS $FLAGS -I$INCLUDE -o "../bin/$BASE"
