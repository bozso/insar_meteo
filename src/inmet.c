#include <stdio.h>

#include "File.h"
#include "common.h"


int main()
{
    File *file = open("asd.txt", "w");
    
    if (not file)
        return 1;
    
    write(file, "asd\n");

    del(file);
    
    return 0;
}
