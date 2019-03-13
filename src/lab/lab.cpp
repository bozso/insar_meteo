#include <string>
#include <lab/lab.hpp>

using std::string;

typedef DataFile DT;

void DataFile::open()
{
    auto file = this->file();
    
    if ((file = fopen(this->datapath, this->iomode)) == NULL)
    {
        throw;
    }

    this->mem.alloc(sizeof(memtype) * this->recsize
                    + (sizeof(DataFile::idx) + sizeof(dtype)) * ntypes);
    
    this->buffer = this->mem.get();
    this->offsets = this->mem.offset<DT::idx>(this->recsize);
    this->dtypes = this->mem.offset<int>(this->recsize
                                         + ntypes * sizeof(DT::idx));
}


void DataFile::read_rec()
{
    auto file = this->file();
    
    if (fread(this->buffer, this->recsize, 1, file) <= 0)
    {
        throw;
    }
    
    this->nio++;
}

template<class T>
void DataFile::write_rec(T* obj)
{
    auto file = this->file();
    
    if (fwrite(obj, sizeof(T), 1, file) <= 0)
    {
        throw;
    }
    
    this->nio++;
}


void DataFile::close()
{
    auto file = this->file();
    if (file)
    {
        fclose(file);
        file = NULL;
    }
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


extern "C" {

long dtype_size(long type) noexcept
{
    return type < ndtype ? sizes[type] : sizes[0];
}

void dtor_datafile(DataFile* datafile)
{
    printf("Calling DataFile destructor.\n");
    datafile->close();
    dtor_memory(&(datafile->mem));
}

}
