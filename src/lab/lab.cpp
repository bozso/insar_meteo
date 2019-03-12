#include <lab/lab.hpp>
#include <lab/aux.hpp>

using std::string;

DataFile::DataFile(string const& datapath, long const recsize,
                   long const ntypes, std::ios_base::openmode iomode) :
iomode(iomode), ntypes(ntypes), recsize(recsize), nio(0)
{
    this->datapath = datapath;
    this->file.open(datapath, this->iomode | std::ios::binary);
    
    
    if (not this->file.is_open())
    {
         throw;
    }
    
    this->mem((sizeof(memtype) + sizeof(DataFile::idx) + sizeof(dtype)) * ntypes)
    
    this->buffer = this->mem.get();
    this->offsets = this->mem.offset<memtype>(ntypes);
    this->dtypes = this->mem.offset<DataFile::idx>(ntypes);
}

void DataFile::readrec()
{
    this->file.read(this->buffer.get(), this->recsize);
    this->nio++;
}


void DataFile::close() { this->file.close(); }

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


string const DataFile::dt2str(int type) noexcept
{
    return type < ndtype ? dtype_names[type] : dtype_names[0];
}

dtype DataFile::str2dt(string const& type) noexcept
{
    for (int ii = 0; ii < ndtype; ++ii)
    {
        if (type == dtype_names[ii])
        {
            return static_cast<dtype>(ii);
        }
    }
    
    return static_cast<dtype>(0);
}


static constexpr int nftype = 5;

static string const ftype_names[nftype] = {
    "Unknown",
    "Array",
    "Matrix",
    "Vector",
    "Records"
};


string const DataFile::ft2str(int type) noexcept
{
    return type < nftype ? ftype_names[type] : ftype_names[0];
}

DataFile::ftype DataFile::str2ft(string const& type) noexcept
{
    for (int ii = 0; ii < nftype; ++ii)
    {
        if (type == ftype_names[ii])
        {
            return static_cast<DataFile::ftype>(ii);
        }
    }
    
    return DataFile::Unknown;
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


long dtype_size(dtype type) noexcept
{
    return type < ndtype ? sizes[type] : sizes[0];
}
