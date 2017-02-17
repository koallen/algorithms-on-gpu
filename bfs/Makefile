all: test

test: main.o bfs.o
	nvcc -o $@ $?

%.o: %.cu
	nvcc -c $<

clean:
	rm -f *.o test
