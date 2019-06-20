#include "aux.hpp"
#include "array.hpp"
#include "type_info.hpp"


namespace aux {

idx const ArrayMeta::operator()(idx const ii) const
{
    return ii * strides[0];
}

idx const ArrayMeta::operator()(idx const ii, idx const jj) const
{
    return ii * strides[0] + jj * strides[jj];
}

idx const ArrayMeta::operator()(idx const ii, idx const jj,
                                idx const kk) const
{
    return ii * strides[0] + jj * strides[jj] + kk * strides[kk];
}

idx const ArrayMeta::operator()(idx const ii, idx const jj, idx const kk,
                                idx const ll) const
{
    return   ii * strides[0] + jj * strides[1] 
           + kk * strides[2] + ll * strides[3];
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

RTypeInfo const& type_info(int const type) noexcept
{
    return type < 16 and type > 0 ? type_infos[type] : type_infos[0];
}


RTypeInfo const& type_info(dtype const type) noexcept
{
    return type_info(static_cast<int const>(type));
}

// end namespace
}