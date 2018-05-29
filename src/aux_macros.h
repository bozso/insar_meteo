#ifndef AUX_MACROS_H
#define AUX_MACROS_H

/*******************************
 * WGS-84 ELLIPSOID PARAMETERS *
 *******************************/

// RADIUS OF EARTH
#define R_earth 6372000

#define WA  6378137.0
#define WB  6356752.3142

// (WA*WA-WB*WB)/WA/WA
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
 * FOR MACROS             *
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
        errorln("Required number of arguments is %d, current number of \
                 \narguments: %d!",\
              (num), argc - min_arg);\
        print_doc(fun_name);\
        return err_arg;\
    }\
})

/*****************************
 * SWITCHING BETWEEN MODULES *
 *****************************/

#define str_isequal(string1, string2) (strcmp((string1), (string2)) == 0)
#define module_select(string) str_isequal(argv[1], string)

/*********************
 * ALLOCATION MACROS *
 *********************/

#define aux_malloc(type, num) (type *) \
                        malloc_or_exit((num) * sizeof(type), __FILE__, __LINE__)
	
#define aux_free(pointer) ({ free(pointer); pointer = NULL; })
	
#define aux_alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define aux_realloc(pointer, type, size) (pointer) = (type *) \
                                     realloc(pointer, (size) * sizeof(type))	

/**************************
 * GSL convenience macros *
 **************************/

#define Mset(matrix, ii, jj, data) gsl_matrix_set((matrix), (ii), (jj), (data))
#define Vset(vector, ii, data) gsl_vector_set((vector), (ii), (data))

#define Mget(matrix, ii, jj) gsl_matrix_get((matrix), (ii), (jj))
#define Vget(vector, ii, data) gsl_vector_get((vector), (ii))

#define Mptr(matrix, ii, jj) gsl_matrix_ptr((matrix), (ii), (jj))
#define Vptr(vector, ii) gsl_vector_ptr((vector), (ii))

/*************
 * IO MACROS *
 *************/

#define error(string) fprintf(stderr, string)
#define errorln(format, ...) fprintf(stderr, format "\n", __VA_ARGS__)

#define println(format, ...) printf(format "\n", __VA_ARGS__)

#define Log printf("%s\t%d\n", __FILE__, __LINE__)

#define aux_read(file, format, ...)\
({\
    if (fscanf(file, format, __VA_ARGS__) <= 0)\
        return 1;\
})

#define aux_fopen(file, path, mode)\
({\
    if ( (file = fopen(path, mode)) == NULL)\
        return err_io;\
})

/***************
 * ERROR CODES *
 ***************/

#define err_io -1
#define err_alloc -2
#define err_num -3
#define err_arg -4

// Idx -- column major order
#define Idx(ii, jj, nrows) (ii) + (jj) * (nrows)

#endif
