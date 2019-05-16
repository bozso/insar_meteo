#include "aux.hpp"


extern "C" {

void* get_memory(size_t const size)
{
    //std::cout << "Trying to allocate memory of size: " << size << aux::end;
    return malloc(size);
}

void release_memory(void* ptr)
{
    free(ptr);
}

}

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


memptr Memory::get() const noexcept { return _memory.get(); }
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
    RTypeInfo::make_info<int>("int"),
    RTypeInfo::make_info<long>("long"),
    RTypeInfo::make_info<size_t>("size_t"),
    
    RTypeInfo::make_info<int8_t>("int8"),
    RTypeInfo::make_info<int16_t>("int16"),
    RTypeInfo::make_info<int32_t>("int32"),
    RTypeInfo::make_info<int64_t>("int64"),

    RTypeInfo::make_info<uint8_t>("uint8"),
    RTypeInfo::make_info<uint16_t>("uint16"),
    RTypeInfo::make_info<uint32_t>("uint32"),
    RTypeInfo::make_info<uint64_t>("uint64"),
    
    RTypeInfo::make_info<float>("float32"),
    RTypeInfo::make_info<double>("float64"),
    
    RTypeInfo::make_info<cpxf>("complex64"),
    RTypeInfo::make_info<cpxd>("complex128")
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
