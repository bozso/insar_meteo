#ifndef INSAR_H
#define INSAR_H

/************************
 * Structs and typedefs *
 ************************/

typedef unsigned int uint;
typedef const double cdouble;

typedef struct { float lon, lat;  } psxy;
typedef struct { int ni; float lon, lat, he, ve; } psxys;
typedef struct { double x, y, z, lat, lon, h; } station; // [m,rad]

typedef struct { double x, y, z; } cart; // Cartesian coordinates
typedef struct { double lon, lat, h; } llh; // Cartesian coordinates

typedef struct {
    uint is_centered, deg;
    double * coeffs, * mean_coords;
    double t_start, t_stop, t_mean;
} orbit_fit;

void azi_inc(cdouble start_t, cdouble stop_t, cdouble mean_t,
             double * coeffs, double * coords, double * mean_coords,
             double * azi_inc, uint is_centered, uint deg, uint max_iter,
             uint is_lonlat, uint ncoords);
#endif
