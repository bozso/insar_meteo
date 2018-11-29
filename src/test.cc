#include "utils.hh"

extern "C" void test(double *arr, size_t shape)
{
    double sum = 0.0;

    FORZ(ii, shape)
        sum += arr[ii] + 1.0;

    //printf("Sum: %lf\n", sum);
}
