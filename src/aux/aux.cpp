#include "aux.hpp"

namespace aux {

static std::unique_ptr<memtype[]> make_memory(long size)
{
    return std::unique_ptr<memtype[]>(new memtype[size]);
}


Memory::Memory(long size): _size(size)
{
    if (size > 0) {
        this->memory = make_memory(size);
    }
    else {
        this->memory = nullptr;
    }
}


void Memory::alloc(long size)
{
    this->_size = size;
    this->memory = make_memory(size);
}


bool is_complex(int type)
{
    switch(type) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
            return false;
        case 15:
        case 16:
            return true;
        default:
            throw std::runtime_error("Unrecognized datatype!");
    }
}

namespace types {

static TypeInfo const Bool = TypeInfo::make_info<bool>();
static TypeInfo const Int  = TypeInfo::make_info<int>();
    //TypeInfo<bool> const Bool    = TypeInfo<bool>();
    //static auto const Int     = TypeInfo<int>();
    //static auto const Long    = TypeInfo<long>();
    //static auto const Size_t  = TypeInfo<size_t>();
    
    //static auto const Int8    = TypeInfo<int8_t>();
    //static auto const Int16   = TypeInfo<int16_t>();
    //static auto const Int32   = TypeInfo<int32_t>();
    //static auto const Int64   = TypeInfo<int64_t>();

// types namespace
}


// return reference to type_info struct
TypeInfo const& type_info(int const type)
{
    switch(static_cast<dtype>(type)) {
        case dtype::Bool:
            return types::Bool;
        case dtype::Int:
            return types::Int;
        default:
            return TypeInfo(0);
    }
}


TypeInfo const& type_info(dtype const type)
{
    return type_info(static_cast<int const>(type));
}

// aux namespace
}
