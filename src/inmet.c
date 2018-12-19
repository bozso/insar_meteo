#include <stdio.h>
#include "File.h"

int main()
{
    File file = open("asd.txt", "w");
    
    write(&file, "blah");
    
    decref(file);
    
    return 0;
}
