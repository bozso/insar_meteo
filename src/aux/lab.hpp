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
    
    DataFile(string const& name, iomode const mode);
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


DataFile::DataFile(string const& name, iomode const mode)
{
    auto sec = ws.inifile[name];
    
    this->datapath = sec["datapath"].as<string>();
    
    char *mode_ = NULL;
    
    switch (mode)
    {
        case DataFile::read:
            mode_ = "rb";
            break;

        case DataFile::write:
            mode_ = "wb";
            break;
    }
    

    if ((this->file = fopen(this->datapath.c_str(), mode_)) == NULL)
    {
         throw;
    }
    
    
}

void DataFile::close()
{
    fclose(this->file);
    this->is_closed = true;
}


DataFile::~DataFile()
{
    if (not this->is_closed)
    {
        this->close();
    }
}


#endif

// guard
#endif
