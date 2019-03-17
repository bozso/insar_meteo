#include "numpy.hpp"

using Array::idx;


int Array::check_type(dtype const type) const
{
    if (static_cast<dtype>(this->type) != type))
    {
        // error
        return true;
    }
    return false;
}


int Array::check_ndim(idx const ndim) const
{
    if (this->ndim != ndim)
    {
        // error
        return true;
    }
    return false;
}


int Array::check_rows(idx const rows) const
{
    if (shape[0] != rows)
    {
        // error
        return true;
    }
    return false;
}


int Array::check_cols(idx const cols) const
{
    if (shape[1] != cols)
    {
        // error
        return true;
    }
    return false;
}
