#include <stdio.h>

#include "File.h"
#include "utils.h"
#include "common.h"


int main()
{
    File *file = open("asd.txt", "r");
    int num = 0;
    
    if (not file)
        return 1;
    
    read(file, "%d\n", &num);
    printf("%d\n", num);
    
    

    del(file);
    
    return 0;
}
