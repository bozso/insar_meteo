inmet_src = $(wildcard *.c) $(wildcard aux/*.c)
inmet_obj = $(inmet_src:.c=.o)


CC = gcc
FLAGS = -O0
#FLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lm
INC_INMET = ./include
LIBS = /home/istvan/miniconda3/lib

%.o: %.c
	$(CC) $(FLAGS) -o $@ -c $< -I$(INC_INMET)

inmet: $(inmet_obj)
	$(CC) $(FLAGS) -o $@ $^ $(LDFLAGS) -L$(LIBS)
	mv inmet ../bin


.PHONY: clean
clean:
	rm -f $(inmet_obj)

install:
	mv inmet ../bin
