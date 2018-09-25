#include <stdarg.h>
#include <string>

#include "utils.hh"

#define min_arg 2

namespace utils {

bool main_check_narg(const int argc, const char * Modules)
{
    if (argc < 2) {
        error("\nAt least one argument, the module name, is required.\
                 \n\nModules to choose from: %s.\n\n", Modules);
        error("Use --help or -h as the first argument to print the help message.\n");
        return true;
    }
    return false;
}

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


bool open(infile& file, const char * path, const bool binary)
{
    const char * mode;
    
    if (binary)
        mode = "rb";
    else
        mode = "r";
    
    if ((file._file = fopen(path, mode)) == NULL) {
        errorln("Failed to open file %s!", path);
        perror("Error");
        return true;
    }
    return false;
}

bool open(outfile& file, const char * path, const bool binary)
{
    const char * mode;
    
    if (binary)
        mode = "wb";
    else
        mode = "w";
    
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

int print(outfile& file, const char * fmt, ...)
{
    int ret = 0;
    va_list ap;
    
    va_start(ap, fmt);
    ret = vfprintf(file._file, fmt, ap);
    va_end(ap);

    return ret;
}

int scan(infile& file, const char * fmt, ...)
{
    int ret = 0;
    va_list ap;
    
    va_start(ap, fmt);
    ret = vfscanf(file._file, fmt, ap);
    va_end(ap);

    return ret;
}

int read(infile& file, const size_t size, const size_t num, void * ptr)
{
    return fread(ptr, size, num, file._file);
}

int write(outfile& file, const size_t size, const size_t num, void * ptr)
{
    return fwrite(ptr, size, num, file._file);
}

}
