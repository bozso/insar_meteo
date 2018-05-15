#ifndef INSAR_H
#define INSAR_H

typedef unsigned int uint;
typedef const double cdouble;

//-----------------------------------------------------------------------------
// STRUCTS
//-----------------------------------------------------------------------------

typedef struct { float lon, lat;  } psxy;
typedef struct { int ni; float lon, lat, he, ve; } psxys;
typedef struct { double x, y, z, lat, lon, h; } station; // [m,rad]

typedef struct { double x, y, z; } cart; // Cartesian coordinates
typedef struct { double lon, lat, h; } llh; // Cartesian coordinates

typedef struct {
    uint is_centered, deg;
    double * coeffs, *mean_coords;
    double t_start, t_stop, t_mean;
} orbit_fit;

void testfun(double * data, uint nrows, uint ncols);

#endif
