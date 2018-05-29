#include <stdio.h>
#include <aux_macros.h>

int read_fit(char * path)
{
    FILE * fit_file = NULL;
    if ((fit_file = fopen(path, "r")) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("read_fit");
        return Err_Io;
    }
    
    return 0;
}

int main(int argc, char **argv)
{
    
    return 0;
}
