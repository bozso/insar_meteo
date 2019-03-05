#ifndef LAB_HPP
#define LAB_HPP

#include <string>

#include <ini.hpp>

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


void activate(string const& path);
void deactivate(std::string const& path = "0");


#ifdef m_get_impl

struct Workspace {
    ini::IniFile inifile;
    string path;
};


static Workspace ws;

void activate(string const& path)
{
    ws.path = path;
    ws.inifile.load(path);
}


void deactivate(string const& path)
{
    if (path != "0")
    {
        ws.path = path;
    }
    
    ws.inifile.save(ws.path);
}


static string ftype2str(DataFile::ftype filetype)
{
    switch (filetype)
    {
        case array:
            return "array";
        case matrix:
            return "matrix";
        case vector:
            return "vector";
        case records:
            return "records";
        default:
            throw;
    }
}

static DataFile::filetype str2ftype(string const& str)
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



DataFile::DataFile(string const& name)
{
    auto sec = ws.inifile[name];
    
    this->datapath = sec["datapath"].as<string>();
    
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
