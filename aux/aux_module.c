#include <stdio.h>
#include <stdlib.h>
#include <aux_module.h>

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
