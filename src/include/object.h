#ifndef OBJECT_H
#define OBJECT_H

#include <stddef.h>
#include <stdarg.h>

#define ObjCast(obj) ((Object const*)(obj))

#define str(obj) (ObjCast(obj)->str((obj)))

typedef void* var;

typedef struct _Object {
    size_t size, refcount;
    var (*ctor)(var self, va_list *app);
    var (*dtor)(var self);
    char const* (*str)(var const self);
} Object;

var new(const var _class, ...);

var ref(var _class);
void incref(var _class);
void decref(var _class);

#endif
