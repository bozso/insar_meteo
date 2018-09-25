inmet_src = inmet.cc main_functions.cc aux/satorbit.cc aux/utils.cc
inmet_obj = $(inmet_src:.cc=.o)


CC = gcc
FLAGS = -O0
#FLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lstdc++ -lm
INCLUDE = /home/istvan/miniconda3/include/eigen3
LIBS = /home/istvan/miniconda3/lib

%.o: %.cc
	$(CC) $(FLAGS) -o $@ -c $< -I$(INCLUDE)

inmet: $(inmet_obj)
	$(CC) $(FLAGS) -o $@ $^ $(LDFLAGS) -L$(LIBS)
	mv inmet ../bin

.PHONY: clean
clean:
	rm -f $(inmet_obj)

install:
	mv inmet ../bin
