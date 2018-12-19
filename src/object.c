#include "object.h"
#include <stdlib.h>
#include <assert.h>
#include <iso646.h>


var new(const var _class, ...)
{
    
    const Object *class = _class;
    var p = calloc(1, class->size);
    
    assert(p);
    
    * (const Object **) p = class;
    ((Object *)p)->refcount = 0;
    
    if (class->ctor) {
        va_list ap;
        
        va_start(ap, _class);

        p = class->ctor(p, &ap);
        ((Object *)p)->refcount = 1;
        
        va_end(ap);
    }
    return p;
}


var ref(var _class)
{
    ((Object *)_class)->refcount++;
    return _class;
}


void incref(var _class)
{
    ((Object *)_class)->refcount++;
}


void decref(var self)
{
    Object **cp = self;
    
    if (self and *cp and --((*cp)->refcount) == 0 and (*cp)->dtor) {
        self = (*cp)->dtor(self);
        free(self);
    }
}

