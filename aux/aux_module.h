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

typedef struct {
    uint deg;
    double * coeffs;
    double t_start, t_stop;
} orbit_notcentered;

typedef struct {
    uint deg;
    double *coeffs, *mean_coords;
    double t_start, t_stop, t_mean;
} orbit_centered;

typedef struct {
    uint is_centered;
    union {
        orbit_notcentered orb_nc;
        orbit_centered orb_c;
    } Orbit;
} orbit;

// satellite orbit record
typedef struct { double t, x, y, z; } torb;

void * malloc_or_exit (size_t nbytes, const char * file, int line);
FILE * sfopen (const char * path, const char * mode);
void ell_cart (station *sta);
void cart_ell (station *sta);
void change_ext (char *name, char *ext);

void azim_elev(const station ps, const station sat, double *azi, double *inc);

void poly_sat_pos(station *sat, const double time, const double *poli,
                  const uint deg_poly);

void poly_sat_vel(double *vx, double *vy, double *vz, const double time,
                  const double *poli, const uint deg_poly);

double sat_ps_scalar(station * sat, const station * ps, const double time,
                     const double * poli, const uint poli_deg);

void closest_appr(const double *poli, const size_t pd, const double tfp,
                  const double tlp, const station * ps, station * sat,
                  const uint max_iter);

void axd(const double  a1, const double  a2, const double  a3,
         const double  d1, const double  d2, const double  d3,
         double *n1, double *n2, double *n3);

#define QUOTE(s) # s   /* turn s into string "s" */

/* minimum number of arguments:
 *     - argv[0] is the executable name
 *     - argv[1] is the module name
 */
#define Min_Arg 2

//----------------------------------------------------------------------
// WGS-84 ELLIPSOID PARAMETERS
//----------------------------------------------------------------------

// RADIUS OF EARTH
#define R_earth 6372000

#define WA  6378137.0
#define WB  6356752.3142

// (WA*WA-WB*WB)/WA/WA
#define E2  6.694380e-03


//----------------------------------------------------------------------
// DEGREES, RADIANS, PI
//----------------------------------------------------------------------

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01
#define distance(x, y, z) sqrt((y)*(y)+(x)*(x)+(z)*(z))

#define OR ||
#define AND &&

//----------------------------------------------------------------------
// FOR MACROS
// REQUIRES C99 standard!
//----------------------------------------------------------------------

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))


/*----------------------------------------------------------------------
 * Documentation and argument number checking
 *---------------------------------------------------------------------- */

#define Mk_Doc(fun_name, doc) char * fun_name ## __doc__ = doc
#define Print_Doc(fun_name) printf("%s", fun_name ## __doc__)

#define Aux_CheckArg(fun_name, num)\
({\
    if (argc != ((num) + Min_Arg)) {\
        error("Required number of arguments is %d, current number of arguments: %d!",\
              (num), argc - Min_Arg);\
        Print_Doc(fun_name);\
        return(Err_Arg);\
    }\
})

//----------------------------------------------------------------------
// SWITCHING BETWEEN MODULES
//----------------------------------------------------------------------

#define Str_IsEqual(string1, string2) (strcmp((string1), (string2)) == 0)
#define Str_Select(string) Str_IsEqual(argv[1], string)

//----------------------------------------------------------------------
// ALLOCATION MACROS
//----------------------------------------------------------------------

#define Aux_Malloc(type, num) (type *) \
                        malloc_or_exit((num) * sizeof(type), __FILE__, __LINE__)
	
#define Aux_Free(pointer) ({ free(pointer); pointer = NULL; })
	
#define Aux_Alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define Aux_Realloc(pointer, type, size) (pointer) = (type *) \
                                     realloc(pointer, (size) * sizeof(type))	

/*----------------------------------------------------------------------
 * GSL convenience macros
 *---------------------------------------------------------------------- */

#define Mset(matrix, ii, jj, data) gsl_matrix_set((matrix), (ii), (jj), (data))
#define Vset(vector, ii, data) gsl_vector_set((vector), (ii), (data))

#define Mptr(matrix, ii, jj) gsl_matrix_ptr((matrix), (ii), (jj))
#define Vptr(vector, ii) gsl_vector_ptr((vector), (ii))

//----------------------------------------------------------------------
// IO MACROS
//----------------------------------------------------------------------

#define error(format, ...) fprintf(stderr, format, __VA_ARGS__)
#define errorln(format, ...) fprintf(stderr, format"\n", __VA_ARGS__)

#define println(format, ...) printf(format"\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

//----------------------------------------------------------------------
// ERROR CODES
//----------------------------------------------------------------------

#define Err_Io -1
#define Err_Alloc -2
#define Err_Num -3
#define Err_Arg -4

#endif
