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

/*****************************
 * SWITCHING BETWEEN MODULES *
 *****************************/

#define Str_IsEqual(string1, string2) (strcmp((string1), (string2)) == 0)
#define Module_Select(string) Str_IsEqual(argv[1], string)

//----------------------------------------------------------------------
// ALLOCATION MACROS
//----------------------------------------------------------------------

#define Aux_Malloc(type, num) (type *) \
                        malloc_or_exit((num) * sizeof(type), __FILE__, __LINE__)
	
#define Aux_Free(pointer) ({ free(pointer); pointer = NULL; })
	
#define Aux_Alloc(type, num) (type *) calloc((num) * sizeof(type))	
#define Aux_Realloc(pointer, type, size) (pointer) = (type *) \
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

/***************
 * ERROR CODES *
 ***************/

#define Err_Io -1
#define Err_Alloc -2
#define Err_Num -3
#define Err_Arg -4

// Idx -- column major order
#define Idx(ii, jj, nrows) (ii) * (nrows) + (jj)

#endif
