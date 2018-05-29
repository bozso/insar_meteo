#ifndef INSAR_H
#define INSAR_H

/************************
 * Structs and typedefs *
 ************************/

typedef unsigned int uint;
typedef const double cdouble;

typedef struct { double x, y, z; } cart; // Cartesian coordinates
typedef struct { double lon, lat, h; } llh; // Cartesian coordinates

void * malloc_or_exit(size_t nbytes, const char * file, int line);
double norm(cdouble x, cdouble y, cdouble z);

#endif
