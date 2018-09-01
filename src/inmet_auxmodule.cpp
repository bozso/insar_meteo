
#include "array.hpp"

pydoc(test, "test");

static py_ptr test(py_noargs)
{
    print("Test.\n");
    Py_RETURN_NONE;
}

init_methods(inmet_aux,
             pymeth_noargs(test))
             
init_module(inmet_aux, "inmet_aux", 0.1)
