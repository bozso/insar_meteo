#ifndef REFCOUNT_H
#define REFCOUNT_H

#define incref(obj) (obj)->ref.refcount++

#define decref(obj)\
do{\
    if((obj) != NULL) {\
        _decref(&((obj)->ref));\
    }\
} while(0)


#define pydecref(obj)\
do{\
    if((obj) != NULL) {\
        _decref(&((obj)->ref));\
        PY_XDECREF((obj)->pyobj);\
    }\
} while(0)


typedef struct _reference {
    void *ptr;
    unsigned int refcount;
} reference;

void _decref(reference * ref);

#endif
