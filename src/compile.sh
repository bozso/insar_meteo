CC=gcc
FLAGS='-O3 -std=c99'
LIBS='-lm'
BASE=$(basename $1 | cut -d'.' -f1)
echo $BASE
$CC $1 $LIBS $FLAGS -o "../bin/$BASE"
