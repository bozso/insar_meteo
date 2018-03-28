#include <stdio.h>
#include <tgmath.h>

#include "Python.h"
#include "capi_macros.h"

//-----------------------------------------------------------------------------
// STRUCTS
//-----------------------------------------------------------------------------

typedef unsigned int uint;

typedef struct {
    double mean_t, mean_coords[3];
    double start_t, stop_t;
    double * coeffs;
    uint is_centered, deg;
} orbit_fit;

typedef struct { double x, y, z; } cart;

#define GET_LIST_ITEM(list, idx) PyFloat_AsDouble(PyList_GetItem(list, idx))

//-----------------------------------------------------------------------------
// AUXILLIARY FUNCTIONS
//-----------------------------------------------------------------------------

static double norm(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

static FILE * sfopen(const char * path, const char * mode)
{
    FILE * file = fopen(path, mode);

    if (!file) {
        errora("Could not open file \"%s\"   ", path);
        perror("fopen");
        return NULL;
    }
    return(file);
}

static int read_orbit_fit(const char * infile, orbit_fit * orb)
{
    FILE * input = sfopen(infile, "r");
    uint deg, is_centered;
    
    fscanf(input, "%d\n", &is_centered);
    
    if (is_centered) {
        fscanf(input, "%lf %lf %lf %lf\n", &(orb->mean_t),
                                    &(orb->mean_coords[0]),
                                    &(orb->mean_coords[1]),
                                    &(orb->mean_coords[2]));
    }
    else {
        orb->mean_t = 0.0;
        orb->mean_coords[0] = 0.0; orb->mean_coords[1] = 0.0;
        orb->mean_coords[2] = 0.0;
    }
    
    orb->is_centered = is_centered;
    
    fscanf(input, "%lf %lf\n", &(orb->start_t), &(orb->stop_t));
    fscanf(input, "%d\n", &deg);
    
    orb->deg = deg;
    
    orb->coeffs = PyMem_New(double, 3 * (deg + 1));
    
    if (!(orb->coeffs))
        return 1;
    
    FOR(ii, 0, 3 * (deg + 1))
        fscanf(input, "%lf\n", &(orb->coeffs[ii]));
    
    return 0;
}

static void calc_pos(const orbit_fit * orb, double time, cart * pos)
{
    uint n_poly = orb->deg + 1, is_centered = orb->is_centered;
    double x = 0.0, y = 0.0, z = 0.0;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (is_centered)
        time -= orb->mean_t;
    
    if(n_poly == 2) {
        x = coeffs[0] * time + coeffs[1];
        y = coeffs[2] * time + coeffs[3];
        z = coeffs[4] * time + coeffs[5];
    }
    else {
        x = coeffs[0]           * time;
        y = coeffs[n_poly]      * time;
        z = coeffs[2 * n_poly]  * time;

        FOR(ii, 1, n_poly - 1) {
            x = (x + coeffs[             ii]) * time;
            y = (y + coeffs[    n_poly + ii]) * time;
            z = (z + coeffs[2 * n_poly + ii]) * time;
        }
        
        x += coeffs[    n_poly - 1];
        y += coeffs[2 * n_poly - 1];
        z += coeffs[3 * n_poly - 1];
    }
    
    if (is_centered) {
        x += mean_coords[0];
        y += mean_coords[1];
        z += mean_coords[2];
    }
    
    pos->x = x; pos->y = y; pos->z = z;
} // calc_pos

static double dot_product(const orbit_fit * orb, const cart * gnss,
                          double time)
{
    double dx, dy, dz, sat_x = 0.0, sat_y = 0.0, sat_z = 0.0,
                       vel_x, vel_y, vel_z, power, inorm;
    uint n_poly = orb->deg + 1;
    
    const double *coeffs = orb->coeffs, *mean_coords = orb->mean_coords;
    
    if (orb->is_centered)
        time -= orb->mean_t;
    
    // linear case 
    if(n_poly == 2) {
        sat_x = coeffs[0] * time + coeffs[1];
        sat_y = coeffs[2] * time + coeffs[3];
        sat_z = coeffs[4] * time + coeffs[5];
        vel_x = coeffs[0]; vel_y = coeffs[2]; vel_z = coeffs[4];
    }
    // evaluation of polynom with Horner's method
    else {
        
        sat_x = coeffs[0]           * time;
        sat_y = coeffs[n_poly]      * time;
        sat_z = coeffs[2 * n_poly]  * time;

        FOR(ii, 1, n_poly - 1) {
            sat_x = (sat_x + coeffs[             ii]) * time;
            sat_y = (sat_y + coeffs[    n_poly + ii]) * time;
            sat_z = (sat_z + coeffs[2 * n_poly + ii]) * time;
        }
        
        sat_x += coeffs[    n_poly - 1];
        sat_y += coeffs[2 * n_poly - 1];
        sat_z += coeffs[3 * n_poly - 1];
        
        vel_x = coeffs[    n_poly - 2];
        vel_y = coeffs[2 * n_poly - 2];
        vel_z = coeffs[3 * n_poly - 2];
        
        FOR(ii, 0, n_poly - 3) {
            power = (double) n_poly - 1.0 - ii;
            vel_x += ii * coeffs[             ii] * pow(time, power);
            vel_y += ii * coeffs[    n_poly + ii] * pow(time, power);
            vel_z += ii * coeffs[2 * n_poly + ii] * pow(time, power);
        }
    }
    
    if (orb->is_centered) {
        sat_x += mean_coords[0];
        sat_y += mean_coords[1];
        sat_z += mean_coords[2];
    }
    
    // satellite coordinates - GNSS coordinates
    dx = sat_x - gnss->x;
    dy = sat_y - gnss->y;
    dz = sat_z - gnss->z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return(vel_x * dx * inorm + vel_y * dy * inorm + vel_z * dz * inorm);
}

static void closest_appr(const orbit_fit * orb, const cart * gnss,
                         const uint max_iter, cart * sat_pos)
{
    // compute the sat position using closest approche
    
    // first, last and middle time
    double t_start = orb->start_t - 5.0,
           t_stop  = orb->stop_t + 5.0,
           t_middle; 
    
    printf("Start time: %lf, Stop time: %lf  ", t_start, t_stop);
    
    // dot products
    double dot_start, dot_middle = 1.0;

    // iteration counter
    uint itr = 0;
    
    dot_start = dot_product(orb, gnss, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, gnss, t_middle);
        
        // change start for middle
        if ((dot_start * dot_middle) > 0.0) {
            t_start = t_middle;
            dot_start = dot_middle;
        }
        // change  end  for middle
        else
            t_stop = t_middle;

        itr++;
    }
    
    calc_pos(orb, t_middle, sat_pos);
    println("Middle time: %lf [s]", t_middle);
    println("Middle time satellite WGS-84 coordinates (x,y,z) [km]: "
           "(%lf, %lf, %lf)", sat_pos->x / 1e3, sat_pos->y / 1e3,
           sat_pos->z / 1e3);
} // end closest_appr

DOC(azi_inc,
"azim_elev");

FUNCTION_KEYWORDS(azi_inc)
{
    const char * fit_file;
    PyObject * list_of_coords;
    orbit_fit orb;
    cart gnss, sat;
    uint max_iter = 1000;

    // topocentric parameters in PS local system
    double xf, yf, zf, xl, yl, zl, t0, gnss_lon, gnss_lat, azi, inc;
    
    static char * keywords[] = {"fit_file", "gnss_coords", "station_lon",
                                "station_lat", "max_iter", NULL };
    
    PARSE_KEYWORDS(keywords, "sO!dd|I:azi_inc", &fit_file, &PyList_Type,
                   &list_of_coords, &gnss_lon, &gnss_lat, &max_iter);
    
    /*
    static char * keywords[] = {"fit_file", "gnss_coords", "station_lon",
                                "station_lat", "max_iter", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO!dd|I:azi_inc", keywords,
                                     &fit_file, &PyList_Type, &list_of_coords,
                                     &gnss_lon, &gnss_lat, &max_iter))
        return NULL;
    */
    
    gnss.x = GET_LIST_ITEM(list_of_coords, 0);
    gnss.y = GET_LIST_ITEM(list_of_coords, 1);
    gnss.z = GET_LIST_ITEM(list_of_coords, 2);
    
    if (read_orbit_fit(fit_file, &orb))
        return PyErr_NoMemory();
    
    println("%lf %lf %lf", orb.coeffs[0], orb.coeffs[1], orb.coeffs[2]);
    
    closest_appr(&orb, &gnss, max_iter, &sat);
        
    xf = sat.x - gnss.x;
    yf = sat.y - gnss.y;
    zf = sat.z - gnss.z;
    
    gnss_lon *= DEG2RAD;
    gnss_lat *= DEG2RAD;
    
    xl = - sin(gnss_lat) * cos(gnss_lon) * xf
         - sin(gnss_lat) * sin(gnss_lon) * yf + cos(gnss_lat) * zf ;

    yl = - sin(gnss_lon) * xf + cos(gnss_lon) * yf;

    zl = + cos(gnss_lat) * cos(gnss_lon) * xf
         + cos(gnss_lat) * sin(gnss_lon) * yf + sin(gnss_lat) * zf ;

    t0 = norm(xl, yl, zl);

    inc = acos(zl / t0) * RAD2DEG;

    if(xl == 0.0) xl = 0.000000001;

    azi = atan(abs(yl / xl));

    if( (xl < 0.0) && (yl > 0.0) ) azi = M_PI - azi;
    if( (xl < 0.0) && (yl < 0.0) ) azi = M_PI + azi;
    if( (xl > 0.0) && (yl < 0.0) ) azi = 2.0 * M_PI - azi;

    azi *= RAD2DEG;

    if(azi > 180.0)
        azi -= 180.0;
    else
        azi +=180.0;
    
    PyMem_Del(orb.coeffs);
    
    return Py_BuildValue("dd", azi, inc);
} // azim_elev


static PyMethodDef InsarMethods[] = {
    METHOD_KEYWORDS(azi_inc),
    //{"azi_inc", (PyCFunction) refl_azi_inc,
    // METH_VARARGS | METH_KEYWORDS, azi_inc__doc__},
    {NULL, NULL, 0, NULL}
};

DOC(insar_aux,
"INSAR_AUX");

static struct PyModuleDef insarmodule = {
    PyModuleDef_HEAD_INIT, "insar_aux", insar_aux__doc__, -1, InsarMethods
};

PyMODINIT_FUNC
PyInit_insar_aux(void)
{
    return PyModule_Create(&insarmodule);
}
