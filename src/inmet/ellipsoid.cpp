#include "aux.hpp"


using aux::ptr;

struct Ellipsoid {
    double a, b, e2;
    
    Ellipsoid() = default;
    ~Ellipsoid() = default;
};


static auto ell = Ellipsoid{0.0, 0.0, 0.0};


Ellipsoid const& get_ellipsoid()
{
    return ell;
}

extern "C" {

void print_ellipsoid()
{
    printf("Ellipsoid in use: a: %lf b: %lf c: %lf\n", ell.a, ell.b, ell.e2);
}


void set_ellipsoid(Ellipsoid const& new_ell)
{
    ell = new_ell;
}

}