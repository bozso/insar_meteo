#include <string>
#include <lab/lab.hpp>

using std::string;

typedef DataFile DT;

error::error(string& format,  ...) : std::exception()
{
    va_list args;
    va_start(args, format);
    vsnprintf(error_msg, m_max_error_len, format.c_str(), args);
    va_end(args);
}

error::error(string&& format, ...) : std::exception()
{
    va_list args;
    va_start(args, format);
    vsnprintf(error_msg, m_max_error_len, format.c_str(), args);
    va_end(args);
}

memptr endswap(memptr objp, long size)
{
    std::reverse(objp, objp + size);
    return objp;
}


memptr noswap(memptr objp, long size)
{
    return objp;
}

DT::DataFile(FileInfo* _info, iomode mode) : info(*_info)
{
    open(mode);
}

void DT::open(iomode mode)
{
    file.open(info.path, mode);
    
    if (not file.is_open())
    {
        throw error("Could not open file: %s!", info.path);
    }
    
    if (info.endswap)
    {
        sfun = endswap;
    }
    else
    {
        sfun = noswap;
    }
    
    mem.alloc(sizeof(memtype) * info.recsize);
    buffer = mem.get();
}


void DataFile::read_rec()
{
    file.read(reinterpret_cast<memptr>(buffer), info.recsize);
    nio++;
}


constexpr int ndtype = 17;

static string const dtype_names[17] = {
    "Unknown",
    "Bool",
    "Int",
    "Long",
    "Size_t",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Complex64",
    "Complex128"
};


string const DT::dt2str(int type) noexcept
{
    return type < ndtype ? dtype_names[type] : dtype_names[0];
}

dtype DT::str2dt(string const& type) noexcept
{
    for (int ii = 0; ii < ndtype; ++ii)
    {
        if (type == dtype_names[ii])
        {
            return static_cast<dtype>(ii);
        }
    }
    
    return dtype::Unknown;
}


static constexpr int nftype = 3;

static string const ftype_names[nftype] = {
    "Unknown",
    "Array",
    "Records"
};


string const DataFile::ft2str(int type) noexcept
{
    return type < nftype ? ftype_names[type] : ftype_names[0];
}

ftype DT::str2ft(string const& type) noexcept
{
    for (int ii = 0; ii < nftype; ++ii)
    {
        if (type == ftype_names[ii])
        {
            return static_cast<ftype>(ii);
        }
    }
    
    return ftype::Unknown;
}


static long const sizes[ndtype] = {
    0,
    sizeof(bool),
    sizeof(int),
    sizeof(long),
    sizeof(size_t),
    sizeof(int8_t),
    sizeof(int16_t),
    sizeof(int32_t),
    sizeof(int64_t),
    sizeof(uint8_t),
    sizeof(uint16_t),
    sizeof(uint32_t),
    sizeof(uint64_t),
    sizeof(float),
    sizeof(double),
    sizeof(cpx64),
    sizeof(cpx128)
};


extern "C" {

long dtype_size(long type) noexcept
{
    return type < ndtype ? sizes[type] : sizes[0];
}

}
