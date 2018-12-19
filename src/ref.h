#ifndef REF_H
#define REF_H

typedef struct _ref {
    void *obj;
    void (*free)(void *);
    int count;
} ref;

void
_incref(const ref *refc);

void
_decref(const ref *refc);

#define incref(obj) _incref((obj).refc)
#define decref(obj) _decref((obj).refc)
    

#endif

