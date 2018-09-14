#include "utils.hpp"

bool open(File& file, const char * path, const char * mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("Error");
        return true;
    }
    return false;
}

