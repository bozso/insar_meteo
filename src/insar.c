#include <stdio.h>
#include <tgmath.h>
#include <stdlib.h>

#include "insar.h"
#include "aux_macros.h"

/* Iterating over array   
 * for(uint jj = 0; jj < ncols; jj++)
 *      for(uint ii = 0; ii < nrows; ii++)
 *          data[Idx(ii, jj, nrows)] = ...;
 */


/************************
 * Auxilliary functions *
 * **********************/


void * malloc_or_exit(size_t nbytes, const char * file, int line)
{
    /* Extended malloc function. */	
    void *x;
	
    if ((x = malloc(nbytes)) == NULL) {
        errorln("%s:line %d: malloc() of %zu bytes failed",
                file, line, nbytes);
        exit(Err_Alloc);
    }
    else
        return x;
}

double norm(cdouble x, cdouble y, cdouble z)
{
    /* Vector norm. */
    return sqrt(x * x + y * y + z * z);
}
