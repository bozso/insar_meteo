inmet_src = $(wildcard *.cc) $(wildcard aux/*.cc) $(wildcard eigen/*.cc)
inmet_obj = $(inmet_src:.cc=.o)


CC = gcc
FLAGS = -O0
#FLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lstdc++ -lm
INC_EIG = /home/istvan/miniconda3/include/eigen3
INC_INMET = /home/istvan/progs/insar_meteo/src/include
LIBS = /home/istvan/miniconda3/lib

%.o: %.cc
	$(CC) $(FLAGS) -o $@ -c $< -I$(INC_EIG) -I$(INC_INMET)

inmet: $(inmet_obj)
	$(CC) $(FLAGS) -o $@ $^ $(LDFLAGS) -L$(LIBS)
	mv inmet ../bin


.PHONY: clean
clean:
	rm -f $(inmet_obj)

install:
	mv inmet ../bin
