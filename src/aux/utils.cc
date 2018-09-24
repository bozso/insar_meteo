#include <stdarg.h>
#include <string>

#include "utils.hh"

#define min_arg 2

using namespace utils;

bool check_narg(const argparse& ap, int req_arg)
{
    std::string first_arg(ap.argv[2]);
    if (first_arg == "-h" or first_arg == "--help") {
        ap.print_usage();
        return true;
    }
    else if (ap.argc != (req_arg + min_arg)) {
        errorln("\n Required number of arguments is %d, current number of "
                "arguments: %d!\n", req_arg, ap.argc - min_arg);
        ap.print_usage();
        return true;
    }
    return false;
};


bool open(File& file, const char * path, const char * mode)
{
    if ((file._file = fopen(path, mode)) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("Error");
        return true;
    }
    return false;
}

void println(char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vprintf(fmt, ap), puts("\n");
    va_end(ap);
}

void errorln(const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap), fputs("\n", stderr);
    va_end(ap);
}

void error(const char * fmt, ...)
{
    va_list ap;
    
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

int fprint(File& file, const char * fmt, ...)
{
    int ret = 0;
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    ret = vfprintf(tmp, fmt, ap);
    va_end(ap);

    return ret;
}

int fscan(File& file, const char * fmt, ...)
{
    int ret = 0;
    FILE * tmp = file._file;
    va_list ap;
    
    va_start(ap, fmt);
    
    ret = vfscanf(tmp, fmt, ap);
    va_end(ap);
    return ret;
}
