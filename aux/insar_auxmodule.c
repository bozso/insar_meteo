#include <stdio.h>
#include <tgmath.h>

#include "Python.h"
#include "capi_macros.h"
#include "numpy/arrayobject.h"

//-----------------------------------------------------------------------------
// STRUCTS
//-----------------------------------------------------------------------------

typedef unsigned int uint;

typedef struct {
    double mean_t;
    double mean_coords[3];
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

static double dot_product(const orbit_fit * orb, const double X, const double Y,
                          const double Z, double time)
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
    dx = sat_x - X;
    dy = sat_y - Y;
    dz = sat_z - Z;
    
    // product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vel_x, vel_y, vel_z));
    
    return(vel_x * dx * inorm + vel_y * dy * inorm + vel_z * dz * inorm);
}

static void closest_appr(const orbit_fit * orb, const double X, const double Y,
                         const double Z, const uint max_iter, cart * sat_pos)
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
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while( fabs(dot_middle) > 1.0e-11 && itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, X, Y, Z, t_middle);
        
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

PyFun_Doc(azi_inc, "azim_elev");

PyFun_Varargs(azi_inc)
{
    double start_t, stop_t, mean_t;
    PyObject *coeffs, *coords, *lonlats, *mean_coords;
    NPY_AO *a_coeffs, *a_coords, *a_lonlats, *a_meancoords;
    uint is_centered, deg, max_iter;

    cart sat;
    orbit_fit orb;
    npy_intp n_coords, n_lonlats, row_coeffs, azi_inc_shape[2];

    // topocentric parameters in PS local system
    double xf, yf, zf, xl, yl, zl, X, Y, Z, t0, lon, lat, azi, inc;
    
    /*
    static char * keywords[] = {"coeffs", "start_t", "stop_t", "mean_t",
                                "mean_coords", "is_centered", "deg" "coords","lonlat", 
                                "max_iter", NULL };
    */
    
    PyFun_Parse_Varargs("OdddOIIOOI:azi_inc", &coeffs, &start_t, &stop_t,
                        &mean_t, &mean_coords, &is_centered, &deg, &coords,
                        &lonlats, &max_iter);
    
    a_coeffs = (NPY_AO *) PyArray_FROM_OTF(coeffs, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!a_coeffs) goto fail;

    NPY_Ndim_Check(a_coeffs, 2); NPY_Dim_Check(a_coeffs, 0, 3);
    NPY_Dim_Check(a_coeffs, 1, deg + 1);
    
    a_coords = (NPY_AO *) PyArray_FROM_OTF(coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!a_coords) goto fail;

    a_lonlats = (NPY_AO *) PyArray_FROM_OTF(lonlats, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!a_lonlats) goto fail;
    
    NPY_Ndim_Check(a_coords, 2); NPY_Ndim_Check(a_lonlats, 2);
    
    n_coords = NPY_Dim(a_coords, 0);
    n_lonlats = NPY_Dim(a_lonlats, 0);
    
    if (n_coords != n_lonlats) {
        PyErr_Format(PyExc_TypeError, "coords (rows: %d) and lonlats (rows: %d)"
                     " do not have the same number of rows\n", n_coords,
                     n_lonlats);
        goto fail;
    }

    
    a_meancoords = (NPY_AO *) PyArray_FROM_OTF(mean_coords, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!a_meancoords) goto fail;
    
    NPY_Ndim_Check(a_meancoords, 1);
    NPY_Dim_Check(a_meancoords, 0, 3);
    
    azi_inc_shape[0] = n_coords;
    azi_inc_shape[1] = 2;
    
    NPY_AO * azi_inc = (NPY_AO *) PyArray_EMPTY(2, azi_inc_shape, NPY_DOUBLE, 0);
    if (!azi_inc) goto fail;
    
    orb.coeffs = (double *) NPY_Data(a_coeffs);
    orb.deg = deg;
    orb.is_centered = is_centered;
    
    orb.start_t = start_t;
    orb.stop_t = stop_t;
    orb.mean_t = mean_t;
    
    orb.mean_coords = (double *) NPY_Data(a_meancoords);
    
    //println("%lf %lf %lf", orb.coeffs[0], orb.coeffs[1], orb.coeffs[2]);
    
    FOR(ii, 0, n_coords) {
        X = NPY_Delem(a_coords, ii, 0);
        Y = NPY_Delem(a_coords, ii, 1);
        Z = NPY_Delem(a_coords, ii, 2);
        
        closest_appr(&orb, X, Y, Z, max_iter, &sat);
            
        xf = sat.x - X;
        yf = sat.y - Y;
        zf = sat.z - Z;
        
        lon = NPY_Delem(a_lonlats, ii, 0) * DEG2RAD;
        lat = NPY_Delem(a_lonlats, ii, 1) * DEG2RAD;
        
        xl = - sin(lat) * cos(lon) * xf
             - sin(lat) * sin(lon) * yf + cos(lat) * zf ;
    
        yl = - sin(lon) * xf + cos(lon) * yf;
    
        zl = + cos(lat) * cos(lon) * xf
             + cos(lat) * sin(lon) * yf + sin(lat) * zf ;
    
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
        
        NPY_Delem(azi_inc, ii, 0) = azi;
        NPY_Delem(azi_inc, ii, 1) = inc;
        
    }

    Py_DECREF(a_coeffs);
    Py_DECREF(a_coords);
    Py_DECREF(a_lonlats);
    return Py_BuildValue("O", azi_inc);

fail:
    Py_XDECREF(a_coeffs);
    Py_XDECREF(a_coords);
    Py_XDECREF(a_lonlats);
    Py_XDECREF(azi_inc);
    return NULL;

} // azim_elev

PyFun_Doc(asc_dsc_select,
"asc_dsc_select");

PyFun_Keywords(asc_dsc_select)
{
    PyObject *in_arr1 = NULL, *in_arr2 = NULL;
    NPY_AO *arr1 = NULL, *arr2 = NULL;
    npy_double max_sep = 100.0;
    
    npy_intp n_arr1, n_arr2;
    uint n_found = 0;
    npy_double lon1, lat1, lon2, lat2, dlon, dlat;
    
    char * keywords[] = {"asc", "dsc", "max_sep", NULL};
    
    PyFun_Parse_Keywords(keywords, "OO|d:asc_dsc_select", &in_arr1, &in_arr2,
                                                          &max_sep);
    
    print("Maximum separation: %6.3lf meters => ", max_sep);
    
    max_sep /= R_earth;
    max_sep = max_sep * max_sep * RAD2DEG * RAD2DEG;
    
    println("approx. %E degrees", max_sep);
    
    arr1 = (NPY_AO *) PyArray_FROM_OTF(in_arr1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!arr1) goto fail;
    
    arr2 = (NPY_AO *) PyArray_FROM_OTF(in_arr2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(!arr2) goto fail;
    
    n_arr1 = NPY_Dim(arr1, 0);
    n_arr2 = NPY_Dim(arr2, 0);
    
    NPY_AO * idx = (NPY_AO *) PyArray_ZEROS(1, &n_arr1, NPY_BOOL, 0);
    
    FOR(ii, 0, n_arr1) {
        lon1 = NPY_Delem(arr1, ii, 0);
        lat1 = NPY_Delem(arr1, ii, 1);

        FOR(jj, 0, n_arr2) {
            lon2 = NPY_Delem(arr2, jj, 0);
            lat2 = NPY_Delem(arr2, jj, 1);
            
            dlon = lon1 - lon2;
            dlat = lat1 - lat2;
            
            if ( dlon * dlon + dlat * dlat < max_sep) {
                *( (npy_bool *) NPY_Ptr1(idx, ii) ) = 1;
                n_found++;
                break;
            }
        }
    }
    
    
    Py_DECREF(arr1);
    Py_DECREF(arr2);
    return Py_BuildValue("OI", idx, n_found);

fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    return NULL;
}

static PyMethodDef InsarMethods[] = {
    PyFun_Method_Varargs(azi_inc),
    PyFun_Method_Keywords(asc_dsc_select),
    {NULL, NULL, 0, NULL}
};

PyFun_Doc(insar_aux,
"INSAR_AUX");

static struct PyModuleDef insarmodule = {
    PyModuleDef_HEAD_INIT, "insar_aux", insar_aux__doc__, -1, InsarMethods
};

PyMODINIT_FUNC
PyInit_insar_aux(void)
{
    import_array();
    return PyModule_Create(&insarmodule);
}
