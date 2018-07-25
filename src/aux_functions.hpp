#ifndef AUX_FUN_H
#define AUX_FUN_H

#include <fstream>
#include <stdexcept>
#include <sstream>

using namespace std;

/***********************
 * Templated functions *
 ***********************/

template<typename T>
T distance(const T& x, const T& y, const T& z)
{
    return sqrt(x * x + y * y + z * z);
}

template<typename T>
T get_cmd_arg(const char * arg)
{
    istringstream ins(arg);
    T out;

    if (!(ins >> out)) {
        cerr << "Argument is \"" << arg << "\"" << endl;
        throw runtime_error("Invalid argument!");
    } else if (!ins.eof()) {
        cerr << "Argument is \"" << arg << "\"" << endl;
        throw runtime_error("Trailing characters after argument!");
    }
    return out;
}

template<typename T>
T get_parameter(ifstream& file, const char * key_char,
                          const char * delim = ":")
{
    T ret;
    string str, key = string(key_char), del = string(delim);
    file >> str >> ret;
    
    if (str == key + del) {
        return ret;
    }
    else {
        cerr << "Could not find keyword \"" << key << "\"\n";
        throw runtime_error("Keyword error!");
    }
    
}

#if 0

template<typename T>
T read_csv(const char * infile)
{
    
    
    
}


// read write sparse matrix

template<typename MatrixType>
void save(const char *filename, const MatrixType& m) const
{
    ofstream f(filename, ios::binary);
    f.write((char *)&m.rows(), sizeof(m.rows()));
    f.write((char *)&m.cols(), sizeof(m.cols()));
    f.write((char *)&m.data(), sizeof(typename MatrixType::Scalar)*m.cols()*m.cols());
    
    f.close();
}

template<typename MatrixType>
void load(const char *filename, MatrixType& m)
{
    typename MatrixType::Index rows, cols;
    ifstream f(filename, ios::binary);
    
    f.read((char *)&rows, sizeof(rows));
    f.read((char *)&cols, sizeof(cols));
    
    m.resize(rows, cols);
    
    f.read((char *)&m.data(), sizeof(typename MatrixType::Scalar) * rows * cols);
    
    if (f.bad())
        throw runtime_error("Error reading matrix!");
    
    f.close();
}

#endif

#endif
