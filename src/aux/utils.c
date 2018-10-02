#include <stdarg.h>

#include "utils.hh"

#define min_arg 2

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

bool check_narg(const argparse *ap, int req_arg)
{
    const char *first_arg = ap->argv[2];
    if (not (strcmp(first_arg, "-h") and strcmp(first_arg, "--help"))) {
        print_usage(ap);
        return true;
    }
    else if (ap->argc != (req_arg + min_arg)) {
        errorln("\n Required number of arguments is %d, current number of "
                "arguments: %d!\n", req_arg, ap->argc - min_arg);
        print_usage(ap);
        return true;
    }
    return false;
};

bool get_arg(const argparse *ap, const uint idx, const char * fmt, void *target)
{
    if (sscanf(ap->argv[1 + idx], fmt, target) != 1) {
        errorln("Invalid argument: %s", ap->argv[1 + idx]);
        print_usage(ap);
        return true;
    }
    return false;
}

bool open(FILE **file, const char * path, const char * mode)
{
    FILE *tmp;
    if ((tmp = fopen(path, mode)) == NULL) {
        errorln("");
        perror("open");
        return true;
    }
    *file = tmp;
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
