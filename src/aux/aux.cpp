#include "aux.hpp"

namespace aux {

/*
static std::unique_ptr<memtype[]> make_memory(long size)
{
    return std::unique_ptr<memtype[]>(new memtype[size]);
}
*/

Memory::Memory(idx const size): _size(size) { _memory = new memtype[size]; }
Memory::~Memory() { delete[] _memory; }

void Memory::alloc(long size)
{
    this->_size = size;
    delete[] this->_memory;
    this->_memory = new memtype[size];
}


memptr Memory::get() const noexcept { return _memory; }
long Memory::size() const noexcept { return _size; }


bool ArrayInfo::check_ndim(idx const ndim) const
{
    auto const& _ndim = this->ndim;
    
    if (ndim != _ndim) {
        fprintf(stderr, "Expected array with %ld dimensions, got: "
                        "%ld dimensional array!\n", ndim, _ndim);
        return true;
    }
    return false;
}


bool ArrayInfo::check_shape(idx const nelem, idx const dim) const
{
    if (nelem != this->shape[dim]) {
        return true;
    }
    return false;
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
