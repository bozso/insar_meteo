inmet_src = $(wildcard *.cc) $(wildcard aux/*.cc)
inmet_obj = $(inmet_src:.cc=.o)


CC = gcc
FLAGS = -O0 -std=c++03
#FLAGS = -O3 -march=native -funroll-loops
LDFLAGS = -lm -lstdc++ -lgsl -lgslcblas
INC_INMET = ./include
INC_GSL = /home/istvan/miniconda3/include
LIBS = /home/istvan/miniconda3/lib

%.o: %.cc
	$(CC) $(FLAGS) -o $@ -c $< -I$(INC_INMET) -I$(INC_GSL)

inmet: $(inmet_obj)
	$(CC) $(FLAGS) -o $@ $^ -L$(LIBS) $(LDFLAGS) 
	mv inmet ../bin


.PHONY: clean
clean:
	rm -f $(inmet_obj)

install:
	mv inmet ../bin
