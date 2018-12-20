#include "refcount.h"

void _decref(reference *ref)
{
    if (--(ref->refcount) == 0) {
        free(ref->ptr);
    }
}
