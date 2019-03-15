#include <iostream>
#include <exception>

#include "lab/lab.hpp"

using DT = DataFile;
using std::cout;
using std::cerr;


extern "C" {
    int test(FileInfo* info)
    {
        try
        {
            DataFile a(info, std::ios::out | std::ios::binary);
        }
        catch(std::exception& e)
        {
            cerr << "Exception caught: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }
}
