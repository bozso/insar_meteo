module inmet.array;

import std.stdint: intptr_t, uintptr_t;


alias Py_intptr_t = intptr_t;
alias Py_uintptr_t = uintptr_t;

/*
typedef struct _object {
    _PyObject_HEAD_EXTRA
    Py_ssize_t ob_refcnt;
    struct _typeobject *ob_type;
} PyObject;
*/

struct PyArrayInterface {
    int two;              /* contains the integer 2 -- simple sanity check */
    int nd;               /* number of dimensions */
    char typekind;        /* kind in array --- character code of typestr */
    int itemsize;         /* size of each element */
    int flags;            /* flags indicating how the data should be 
                             interpreted must set ARR_HAS_DESCR bit to 
                             validate descr*/
    Py_intptr_t *shape;   /* A length-nd array of shape information */
    Py_intptr_t *strides; /* A length-nd array of stride information */
    void *data;           /* A pointer to the first element of the array */
    /* ignored for now
    PyObject *descr;         NULL or data-description (same as descr key
                             of __array_interface__) -- must set ARR_HAS_DESCR
                             flag or this will be ignored. */
};

