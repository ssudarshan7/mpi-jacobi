all: 
	mpicxx -o pjacobi jacobi_doubles.cpp jacobi.h



clean:
	rm -f *.o pjacobi