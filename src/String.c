#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "String.h"


static var string_ctor(var _self, va_list *app)
{
    string *self = _self;
    
    char const *txt = va_arg(*app, char const*);
    
    self->len = strlen(txt);
    self->str = malloc(self->len + 1);
    assert(self->str);
    strcpy(self->str, txt);
    return self;
}

static var string_dtor(var _self)
{
    string *self = _self;
    free(self->str);
    self->str = 0;
    self->len = 0;
    return self;
}

static char const* string_str(var const _self)
{
    return ((string const*)_self)->str;
}


static const Object _String = {
    sizeof(string), 0, &string_ctor, &string_dtor, &string_str
};

const var String = &_String;

