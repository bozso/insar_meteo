#ifndef RAII_H
#define RAII_H

typedef void DtorFn(void *);

struct DtorNode
{
    DtorFn *dtor;
    void *object;
    struct DtorNode *next;
}
    
#define raii_init struct DtorNode *_dtor_head = NULL;

#defin raii_ctor(obj, ctor, dtor, ...)\
do {\
    struct DtorNode dtor_##__LINE__;\
    if (ctor(obj, __VA_ARGS__ )) {\
        dtor_##__LINE__.dtor = (DtorFn*)dtor;\
        dtor_##__LINE__.object = obj;\
        dtor_##__LINE__.next = _dtor_head;\
        _dtor_head = &dtor_##__LINE__;\
    }\
    else {\
        goto fail;\
    }\
} while(0)

#define raii_free\
do {\
    while (_dtor_head != NULL) {
        _dtor_head->dtor(_dtor_head->object);
        _dtor_head = dtorHead__->next;
    }\
} while(0)

#endif
