#ifndef OBJECT_H
#define OBJECT_H

#include <stddef.h>
#include <stdarg.h>

typedef void* var;

typedef struct _object {
    size_t size, refcount;
    var (*ctor)(var self, va_list *app);
    var (*dtor)(var self);
} object;

var new(const var _class, ...);

var ref(var _class);
void incref(var _class);
void decref(var _class);

#endif
