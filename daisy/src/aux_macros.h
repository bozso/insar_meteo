#ifndef AUX_MACROS_H
#define AUX_MACROS_H

#define QUOTE(s) # s   /* turn s into string "s" */
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

//----------------------------------------------------------------------
// IO MACROS
//----------------------------------------------------------------------

#define error(text, ...) error(stderr, text, __VA_ARGS__)
#define errorln(text, ...) error(stderr, text"\n", __VA_ARGS__)

#define println(format, ...) printf(format"\n", __VA_ARGS__)

#define Log print("%s\t%d\n", __FILE__, __LINE__)

//----------------------------------------------------------------------
// ERROR CODES
//----------------------------------------------------------------------

#define IO_ERR    -1
#define ALLOC_ERR -2
#define NUM_ERR   -3

#endif
