#ifndef AUX_MODULE_H
#define AUX_MODULE_H

void * malloc_or_exit(size_t nbytes, const char * file, int line);
FILE * sfopen(const char * path, const char * mode);

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
 *----------------------------------------------------------------------
 */

#define Mk_Doc(fun_name, doc) char * fun_name ## __doc__ = doc
#define Print_Doc(fun_name) printf("%s", fun_name ## __doc__)

#define Aux_CheckArg(fun_name, num)\
({\
    if (argc < (num)) {\
        error("Required number of arguments is %d, current number of arguments: %d!",\
              (num) - Min_Arg, argc - Min_Arg);\
        Print_Doc(fun_name);\
        return(Err_Arg);\
    }\
})


// Switching between modules
#define Str_IsEqual(string1, string2) (strcmp((string1), (string2)) == 0)
#define Str_Select(string) Str_IsEqual(argv[1], string)

#define Aux_Malloc(pointer, num)  ((pointer) = \
                malloc_or_exit((num) * sizeof *(pointer), __FILE__, __LINE__))	
	
#define Aux_Free(pointer) ({ free(pointer); pointer = NULL; })
	
#define Aux_Alloc(num, type) calloc((num) * sizeof(type))	
#define Aux_Realloc(pointer, size, type) (type *) \
                                     realloc(pointer, (size) * sizeof(type))	


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
