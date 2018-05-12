#ifndef AUX_MODULE_H
#define AUX_MODULE_H

typedef unsigned int uint;

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

// satellite orbit record
typedef struct { double t, x, y, z; } torb;

void * malloc_or_exit (size_t nbytes, const char * file, int line);
FILE * sfopen (const char * path, const char * mode);

void ell_cart (const llh *lonlat, cart * coord);
void cart_ell (const cart * coord, llh * lonlat);
void change_ext (char *name, char *ext);
double norm(double x, double y, double z);

void calc_pos(const orbit_fit * orb, double time, cart * pos);

double dot_product(const orbit_fit * orb, const cart * coord, double time);

void closest_appr(const orbit_fit * orb, const cart * coord,
                  const uint max_iter, cart * sat_pos);

void azi_inc(const orbit_fit * orbit, const cart * coords, uint ndata,
             double *azimuths, double *inclinations);

void poly_sat_vel(double *vx, double *vy, double *vz, const double time,
                  const double *poli, const uint deg_poly);

int fit_orbit(const torb * orbits, const uint ndata, const uint deg,
              const uint is_centered, const char * outfile);

void axd(const double  a1, const double  a2, const double  a3,
         const double  d1, const double  d2, const double  d3,
         double *n1, double *n2, double *n3);

#endif
