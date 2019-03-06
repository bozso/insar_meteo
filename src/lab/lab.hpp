#ifndef LAB_HPP
#define LAB_HPP

#include <string>

using std::string;

struct DataFile {
    enum ftype {
        array, matrix, vector, records
    };
    
    enum iomode { read, write };
    
    enum dtype { Unknown, Bool, Int, Long, Size_t,
                 Int8, Int16, Int32, Int64,
                 UInt8, UInt16, UInt32, UInt64,
                 Float32, Float64, Complex64, Complex128
    };
    
    dtype datatype;
    ftype filetype;
    
    string datapath;
    FILE* file;          
    bool is_closed;
    
    DataFile(string const& name);
    void close();
    ~DataFile();
};


#ifdef m_get_impl

static string ftype2str(DataFile::ftype filetype)
{
    switch (filetype)
    {
        case DataFile::array:
            return "array";
        case DataFile::matrix:
            return "matrix";
        case DataFile::vector:
            return "vector";
        case DataFile::records:
            return "records";
        default:
            throw;
    }
}

static DataFile::ftype str2ftype(string const& str)
{
    if (str == "array")
    {
        return DataFile::array;
    }
    else if (str == "matrix")
    {
        return DataFile::matrix;
    }
    else if (str == "vector")
    {
        return DataFile::vector;
    }
    else if (str == "records")
    {
        return DataFile::records;
    }
    else
    {
        throw;
    }
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
