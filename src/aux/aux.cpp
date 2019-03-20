#include "aux.hpp"

namespace aux {

static std::unique_ptr<memtype[]> make_memory(long size)
{
    return std::unique_ptr<memtype[]>(new memtype[size]);
}

Memory::Memory(idx const size): _size(size) { _memory = make_memory(size); }

void Memory::alloc(long size)
{
    this->_size = size;
    this->_memory = make_memory(size);
}


static RTypeInfo const type_infos[] = {
    RTypeInfo(),
    RTypeInfo::make_info<int>(),
    RTypeInfo::make_info<long>(),
    RTypeInfo::make_info<size_t>(),
    
    RTypeInfo::make_info<int8_t>(),
    RTypeInfo::make_info<int16_t>(),
    RTypeInfo::make_info<int32_t>(),
    RTypeInfo::make_info<int64_t>(),

    RTypeInfo::make_info<uint8_t>(),
    RTypeInfo::make_info<uint16_t>(),
    RTypeInfo::make_info<uint32_t>(),
    RTypeInfo::make_info<uint64_t>(),
    
    RTypeInfo::make_info<float>(),
    RTypeInfo::make_info<double>(),
    
    RTypeInfo::make_info<cpxf>(),
    RTypeInfo::make_info<cpxd>()
};


// return reference to type_info struct

RTypeInfo const& type_info(int const type)
{
    return type < 16 and type > 0 ? type_infos[type] : type_infos[0];
}


RTypeInfo const& type_info(dtype const type)
{
    return type_info(static_cast<int const>(type));
}


// aux namespace
}
