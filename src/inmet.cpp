#include "lab/aux.hpp"
#include "lab/lab.hpp"

#define modules "test, azi_inc"

using DT = DataFile;

#define nrange 22831


static int test(int argc, char **argv)
{
    aux::Timer t;

    FILE* inf = fopen("/media/nas1/Dszekcso/ASC_PS_proc/SLC/20160912.mli", "rb");
    
    auto size = sizeof(float);
    auto arr = aux::uarray<float>(nrange);
    
    double avg = 0.0;
    int n = 0;
    
    for (int ii = 0; ii < 4185; ++ii)
    {
        fread(arr.get(), size, nrange, inf);
        
        for (int jj = 0; jj < nrange; ++jj)
        {
            aux::endswap(arr[jj]);
            avg += static_cast<double>(arr[jj] * arr[jj]);
            avg += static_cast<double>(arr[jj] * arr[jj]);
            avg += static_cast<double>(arr[jj] * arr[jj]);
            n++;
        }
    }
    
    printf("Avg: %lf\n", avg / double(n));
    
    fclose(inf);

    t.report();
    
    return 0;
    
    //activate(argv[1]);
    
    //DataFile test{argv[2]};
    
    //return 0;
}


int main(int argc, char **argv)
{                   
    try
    {
        std::string modname = argv[1];
        
        int argc_ = argc - 2;
        char **argv_ = argv + 2;
        
        if (modname == "test")
        {
            return test(argc_, argv_);
        }
        //else if (modname == "azi_inc")
        //{
            //return azi_inc(argc_, argv_)
        //}
        else
        {
            std::cerr << "Unknown module: " << modname
                      << " Available modules: " << modules << "\n";
            return 1;
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Excaption caught: " << e.what() << "\n";
        return 1;
    }
}
