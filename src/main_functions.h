#ifndef MAIN_FUN_H
#define MAIN_FUN_H

#define min_arg 2

#define norm(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))

#define BUFSIZE 10

typedef unsigned int uint;
typedef const double cdouble;

// Cartesian coordinates
typedef struct cart_struct { double x, y, z; } cart; 

// WGS-84 surface coordinates
typedef struct llh_struct { double lon, lat, h; } llh; 

// Orbit records
typedef struct orbit_struct { double t, x, y, z; } orbit;

// Fitted orbit polynoms structure
typedef struct orbit_fit_struct {
    uint deg, centered;
    double t_min, t_max, t_mean;
    double *coeffs, coords_mean[3];
} orbit_fit;


int fit_orbit(int argc, char **argv);
int azi_inc(int argc, char **argv);
int eval_orbit(int argc, char **argv);

#endif
