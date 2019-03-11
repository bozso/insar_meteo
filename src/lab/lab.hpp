#ifndef LAB_HPP
#define LAB_HPP


#include <string>
#include <complex>

using complex128 = std::complex<double>;
using complex64 = std::complex<float>;

enum dtype {
    Unknown = 0,
    Bool = 1,
    Int = 2,
    Long = 3,
    Size_t = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    UInt8 = 9,
    UInt16 = 10,
    UInt32 = 11,
    UInt64 = 12,
    Float32 = 13,
    Float64 = 14,
    Complex64 = 15,
    Complex128 = 16
};


struct DataFile {
    enum ftype {
        Unknown = 0,
        Array = 1,
        Matrix = 2,
        Vector = 3,
        Records = 4
    };
    
    enum iomode { read, write };
    
    dtype datatype;
    ftype filetype;
    
    std::string datapath;
    FILE* file;          
    bool is_closed;
    
    static std::string const dt2str(int type) noexcept;
    static std::string const ft2str(int type) noexcept;
    
    static dtype str2dt(std::string const& type) noexcept;
    static ftype str2ft(std::string const& type) noexcept;
    
    DataFile(std::string const& name);
    void close();
    ~DataFile();
};


long dtype_size(dtype type) noexcept;

#ifdef m_get_impl

using std::string;

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
    sizeof(complex64),
    sizeof(complex128)
};


long dtype_size(dtype type) noexcept
{
    return type < ndtype ? sizes[type] : sizes[0];
}

DataFile::DataFile(string const& datapath)
{
    this->datapath = datapath;
    
    if ((this->file = fopen(this->datapath.c_str(), "rb")) == NULL)
    {
         throw;
    }
    
    
}


void DataFile::close()
{
    if (this->file != NULL)
    {
        fclose(this->file);
        this->file = NULL;
    }
}


DataFile::~DataFile()
{
    this->close();
}


#endif

// guard
#endif
