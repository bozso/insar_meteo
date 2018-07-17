#ifndef MAIN_FUN_H
#define MAIN_FUN_H

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000.0

#define WA  6378137.0
#define WB  6356752.3142

// (WA^2 - WB^2) / WA^2
#define E2  6.694380e-03

/************************
 * DEGREES, RADIANS, PI *
 ************************/

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define DEG2RAD 1.745329e-02
#define RAD2DEG 5.729578e+01
#define distance(x, y, z) sqrt((y)*(y)+(x)*(x)+(z)*(z))

#define OR ||
#define AND &&

/**************************
 * for macros             *
 * REQUIRES C99 standard! *
 **************************/

#define FOR(ii, min, max) for(uint (ii) = (min); (ii) < (max); (ii)++)
#define FORS(ii, min, max, step) for(uint (ii) = (min); (ii) < (max); (ii) += (step))

/**********************************************
 * Documentation and argument number checking *
 **********************************************/

#define mk_doc(fun_name, doc) char * fun_name ## __doc__ = doc
#define print_doc(fun_name) printf("%s", fun_name ## __doc__)

#define aux_checkarg(fun_name, num)\
({\
    if (argc != ((num) + min_arg)) {\
        errorln("\n Required number of arguments is %d, current number of arguments: %d!",\
                 (num), argc - min_arg);\
        print_doc(fun_name);\
        error = err_arg;\
        goto fail;\
    }\
})

/*****************************
 * Switching between modules *
 *****************************/

#define str_isequal(string1, string2) (strcmp((string1), (string2)) == 0)
#define module_select(string) str_isequal(argv[1], string)

/*********************
 * Allocation macros *
 *********************/

#define aux_malloc(ptr, type, num)\
({\
    if (NULL == ((ptr) = malloc(sizeof(type) * (num))))\
    {\
        fprintf(stderr, "FILE: %s, LINE: %d :: Malloc of %s failed",\
                __FILE__, __LINE__, #ptr);\
        error = err_alloc;\
        goto fail;\
    }\
})
        
#define aux_free(pointer)\
({\
    if ((pointer) != NULL)\
    {\
        free((pointer));\
        pointer = NULL;\
    }\
})

#define aux_alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define aux_realloc(pointer, type, size) (pointer) = (type *) \
                                     realloc(pointer, (size) * sizeof(type))	

/**************************
 * GSL convenience macros *
 **************************/

#define Mset(mtx, ii, jj, data) gsl_matrix_set((mtx), (ii), (jj), (data))
#define Vset(vector, ii, data) gsl_vector_set((vector), (ii), (data))

#define Mget(mtx, ii, jj) gsl_matrix_get((mtx), (ii), (jj))
#define Vget(vector, ii) gsl_vector_get((vector), (ii))

#define Mptr(mtx, ii, jj) gsl_matrix_ptr((mtx), (ii), (jj))
#define Vptr(vector, ii) gsl_vector_ptr((vector), (ii))

/*************
 * IO macros *
 *************/

#define error(string) fprintf(stderr, string)
#define errorln(format, ...) fprintf(stderr, format "\n", __VA_ARGS__)

#define println(format, ...) printf(format "\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

#define aux_open(file, path, mode)\
({\
    if (((file) = fopen((path), (mode))) == NULL)\
    {\
        errorln("FILE: %s, LINE: %d :: Failed to open file %s!", __FILE__, __LINE__, path);\
        perror("Error");\
        error = errno;\
        goto fail;\
    }\
})

#define aux_close(file)\
({\
    if ((file) != NULL) {\
        fclose((file));\
        (file) = NULL;\
    }\
})


#define aux_read(file, format, ...)\
({\
    if (fscanf(file, format, __VA_ARGS__) <= 0) {\
        error = err_io;\
        goto fail;\
    }\
})

/***************
 * Error codes *
 ***************/

typedef enum err_code_t {
    err_succes = 0,
    err_io = -1,
    err_alloc = -2,
    err_num = -3,
    err_arg = -4
} err_code;

struct _errdsc {
    int code;
    char * message;
} errdsc[] = {
    { err_succes, "No error encountered."},
    { err_io, "IO error encountered!" },
    { err_alloc, "Allocation error encountered!" },
    { err_num, "Numerical error encountered!" },
    { err_arg, "Command line argument error encountered!" }
};

// Idx -- column major order
#define Idx(ii, jj, nrows) (ii) + (jj) * (nrows)

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
int test_matrix1(void);
int test_matrix2(void);

#endif
