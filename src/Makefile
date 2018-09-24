main_src = $(wildcard *.cc) $(wildcard aux/*.cc)
main_obj = $(main_src:.cc=.o)

CC = gcc
FLAGS = -O0
LDFLAGS = -lstdc++ -larmadillo -lm
INCLUDE = "/home/istvan/miniconda3/include"
LIBS = "/home/istvan/miniconda3/lib"

%.o: %.cc
	$(CC) -o $@ -c $< $(FLAGS) -I$(INCLUDE)

inmet: $(main_obj)
	$(CC) -o $@ $^ $(FLAGS) $(LDFLAGS) -I$(INCLUDE) -L $(LIBS)

.PHONY: clean
clean:
	rm -f $(main_obj)
