#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <aux_module.h>

/* Extended malloc function */
void * malloc_or_exit(size_t nbytes, const char * file, int line)
{	
    void *x;
	
    if ((x = malloc(nbytes)) == NULL) {
        error("%s:line %d: malloc() of %zu bytes failed\n",
              file, line, nbytes);
        exit(Err_Alloc);
    }
    else
        return x;
}

/* Safely open files. */
FILE * sfopen(const char * path, const char * mode)
{
    FILE * file = fopen(path, mode);

    if (!file) {
        error("Error opening file: \"%s\". ", path);
        perror("fopen");
        exit(Err_Io);
    }
    return file;
}

void ell_cart (station *sta)
{
    // from ellipsoidal to cartesian coordinates

    double lat, lon, n;

    lat = sta->lat;
    lon = sta->lon;
    n = WA / sqrt(1.0 - E2 * sin(lat) * sin(lat));

    sta->x = (              n + sta->h) * cos(lat) * cos(lon);
    sta->y = (              n + sta->h) * cos(lat) * sin(lon);
    sta->z = ( (1.0 - E2) * n + sta->h) * sin(lat);

}
// end of ell_cart

void cart_ell (station *sta)
{
    // from cartesian to ellipsoidal coordinates

    double n, p, o, so, co, x, y, z;

    n = (WA * WA - WB * WB);
    x = sta->x; y = sta->y; z = sta->z;
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    so = sin(o); co = cos(o);
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) );
    so = sin(o); co = cos(o);
    n= WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB);

    sta->lat = o;
    
    o = atan(y/x); if(x < 0.0) o += M_PI;
    sta->lon = o;
    sta->h = p / co - n;

}
// end of cart_ell

void change_ext (char *name, char *ext)
{
    // change the extent of name for ext

    int  ii = 0;
    while( name[ii] != '.' && name[ii] != '\0') ii++;
    name[ii] ='\0';

    sprintf(name, "%s.%s", name, ext);
} // end change_ext

void azim_elev(const station ps, const station sat, double *azi, double *inc)
{
    // topocentric parameters in PS local system
    double xf, yf, zf, xl, yl, zl, t0;

    // cart system
    xf = sat.x - ps.x;
    yf = sat.y - ps.y;
    zf = sat.z - ps.z;

    xl = - sin(ps.lat) * cos(ps.lon) * xf
         - sin(ps.lat) * sin(ps.lon) * yf + cos(ps.lat) * zf ;

    yl = - sin(ps.lon) * xf + cos(ps.lon) * yf;

    zl = + cos(ps.lat) * cos(ps.lon) * xf
         + cos(ps.lat) * sin(ps.lon) * yf + sin(ps.lat) * zf ;

    t0 = distance(xl, yl, zl);

    *inc = acos(zl / t0) * RAD2DEG;

    if(xl == 0.0) xl = 0.000000001;

    *azi = atan(abs(yl / xl));

    if( (xl < 0.0) && (yl > 0.0) ) *azi = M_PI - *azi;
    if( (xl < 0.0) && (yl < 0.0) ) *azi = M_PI + *azi;
    if( (xl > 0.0) && (yl < 0.0) ) *azi = 2.0 * M_PI - *azi;

    *azi *= RAD2DEG;

    if(*azi > 180.0)
        *azi -= 180.0;
    else
        *azi +=180.0;
}
// azim_elev

void poly_sat_pos(station *sat, const double time, const double *poli,
                  const uint deg_poly)
{
    sat->x = sat->y = sat->z = 0.0;

    FOR(ii, 0, deg_poly) {
        sat->x += poli[ii]                * pow(time, (double) ii);
        sat->y += poli[deg_poly + ii]     * pow(time, (double) ii);
        sat->z += poli[2 * deg_poly + ii] * pow(time, (double) ii);
    }
}

void poly_sat_vel(double *vx, double *vy, double *vz, const double time,
                  const double *poli, const uint deg_poly)
{
    *vx = 0.0;
    *vy = 0.0;
    *vz = 0.0;

    FOR(ii, 1, deg_poly) {
        *vx += ii * poli[ii]                * pow(time, (double) ii - 1);
        *vy += ii * poli[deg_poly + ii]     * pow(time, (double) ii - 1);
        *vz += ii * poli[2 * deg_poly + ii] * pow(time, (double) ii - 1);
    }
}

double sat_ps_scalar(station * sat, const station * ps, const double time,
                     const double * poli, const uint poli_deg)
{
    double dx, dy, dz, vx, vy, vz, lv, lps;
    // satellite position
    poly_sat_pos(sat, time, poli, poli_deg);

    dx = sat->x - ps->x;
    dy = sat->y - ps->y;
    dz = sat->z - ps->z;

    lps = distance(dx, dy, dz);

    // satellite velocity
    poly_sat_vel(&vx, &vy, &vz, time, poli, poli_deg);

    lv = distance(vx, vy, vz);

    // normed scalar product of satellite position and velocity vector
    return(  vx / lv * dx / lps
           + vy / lv * dy / lps
           + vz / lv * dz / lps);
}

void closest_appr(const double *poli, const size_t pd, const double tfp,
                  const double tlp, const station * ps, station * sat,
                  const uint max_iter)
{
    // compute the sat position using closest approache
    double tf, tl, tm; // first, last and middle time
    double vs, vm = 1.0; // vectorial products

    uint itr = 0;

    tf = 0.0;
    tl = tlp - tfp;

    vs = sat_ps_scalar(sat, ps, tf, poli, pd);

    while( fabs(vm) > 1.0e-11 && itr < max_iter) {
        tm = (tf + tl) / 2.0;

        vm = sat_ps_scalar(sat, ps, tm, poli, pd);

        if ((vs * vm) > 0.0) {
            // change start for middle
            tf = tm;
            vs = vm;
        }
        else
            // change  end  for middle
            tl = tm;

        itr++;
    }
}
// end closest_appr

void axd(const double  a1, const double  a2, const double  a3,
         const double  d1, const double  d2, const double  d3,
         double *n1, double *n2, double *n3)
{
    // vectorial multiplication a x d
   *n1 = a2 * d3 - a3 * d2;
   *n2 = a3 * d1 - a1 * d3;
   *n3 = a1 * d2 - a2 * d1;
}
