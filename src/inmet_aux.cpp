#include <stdio.h>
#include <string.h>

#include <iostream>


#include "aux.hpp"
#include "lab.hpp"
#include "common.hpp"
#include "tpl_inst.hpp"

//#include "array.hpp"
//#include "satorbit.hpp"


/*
using consts::deg2rad;
using consts::rad2deg;
using consts::pi_per_4;
using consts::R_earth;
*/

#define select_module(module) (strcmp((argv[0]), (module)) == 0)
#define modules "test, azi_inc"

using DT = DataFile;

#define nrange 22831


static void swap4(void *v)
{
    char    in[4], out[4];
    memcpy(in, v, 4);
    out[0] = in[3];
    out[1] = in[2];
    out[2] = in[1];
    out[3] = in[0];
    memcpy(v, out, 4);
}


static int test(int argc, char **argv)
{
    Timer t;

    FILE* inf = fopen("/media/nas1/Dszekcso/ASC_PS_proc/SLC/20160912.mli", "rb");
    
    auto size = sizeof(float);
    auto arr = uarray<float>(nrange);
    
    double avg = 0.0;
    int n = 0;
    
    for (int ii = 0; ii < 4185; ++ii)
    {
        fread(arr.get(), size, nrange, inf);
        
        for (int jj = 0; jj < nrange; ++jj)
        {
            swap4(&arr[jj]);
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

/*
static int ell_to_merc(int argc, char **argv)
{
    // check size, datatype
    if (plon->check_type(np_float64) or plat->check_type(np_float64) or
        pxy->check_type(np_float64))
        return 1;
    
    View<double> lon{plon}, lat{plat}, xy{pxy};
    size_t const rows = lon.shape(0);
    
    if (plon->check_ndim(1) or plat->check_ndim(1) or
        pxy->check_ndim(2) or plat->check_rows(rows))
        return 1;
    
    
    if (isdeg) {
        if (fast) {
            m_forz(ii, rows) {
                double Lon = lon(ii);
                double Lat = lat(ii);
                double tmp = a * deg2rad * (Lon - lon0);
                
                xy(ii, 0) = tmp;
                
                xy(ii, 1) =
                rad2deg * log(tan(pi_per_4 + Lat * deg2rad / 2.0 )) * (tmp / Lon);
            }
        } else {
            m_forz(ii, rows) {
                double Lat = lat(ii);
                xy(ii, 0) = a * deg2rad * (lon(ii) - lon0);
                
                double sin_lat = sin(deg2rad * Lat);
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat), e / 2.0);
                
                xy(ii, 1) = a * (tan((pi_per_4 + Lat / 2.0)) * tmp);
            }
        }
    } else {
        if (fast) {
            m_forz(ii, rows) {
                double Lon = lon(ii);
                double Lat = lat(ii);
                double tmp = a * (Lon - lon0);
                
                xy(ii, 0) = tmp;
                
                xy(ii, 1) =
                rad2deg * log(tan(pi_per_4 + Lat * deg2rad / 2.0)) * (tmp / Lon);
            }
        } else {
            m_forz(ii, rows) {
                double Lat = lat(ii);
                xy(ii, 0) = a * (lon(ii) - lon0);
                
                double sin_lat = sin(Lat);
                double tmp = pow( (1 - e * sin_lat) / (1 + e * sin_lat) , e / 2.0);
                
                xy(ii, 1) = a * (tan((pi_per_4 + Lat / 2.0)) * tmp);
            }
        }
    }
    return 0;
}
// ell_to_merc


static int azi_inc(int argc, char **argv)
{
    View<double> coeffs{pcoeffs}, coords{pcoords}, azi_inc{pazi_inc};
    
    // Set up orbit polynomial structure
    fit_poly orb{mean_t, start_t, stop_t, (double*) pmean->get_data(),
                 coeffs, is_centered, deg};
    
    calc_azi_inc(orb, coords, azi_inc, max_iter, islonlat);
    
    return 0;
}
// azi_inc


int asc_dsc_select(arptr parr1, arptr parr2, arptr pidx, double max_sep,
                   size_t* nfound)
{
    View<double> arr1{parr1}, arr2{parr2};
    View<char> idx{pidx};
    
    max_sep /=  R_earth;
    max_sep = (max_sep * rad2deg) * (max_sep * rad2deg);
    
    *nfound = 0;
    size_t const n1 = arr1.shape(0), n2 = arr2.shape(0);
    
    m_forz(ii, n1)
    {
        m_forz(jj, n2)
        {
            double dlon = arr1(ii, 0) - arr2(jj, 0),
                   dlat = arr1(ii, 1) - arr2(jj, 1);
            
            if ((dlon * dlon + dlat * dlat) < max_sep)
            {
                idx(ii) = 1;
                nfound++;
                break;
            }
        }
    }

    return 0;

}
// asc_dsc_select


static int dominant(int argc, char **argv)
{
    View<double> asc{pasc}, dsc{pdsc};
    vector<bool> asc_selected, dsc_selected;
    vector<double> clustered;
    
    asc_selected.reserve(asc.shape(0));
    dsc_selected.reserve(dsc.shape(0));

    return 0;
}
// dominant
*/


extern "C" {

int inmet_aux(int argc, char **argv) {
    try
    {
        int argc_ = argc - 1;
        char **argv_ = argv + 1;
        
        if (select_module("test"))
        {
            return test(argc_, argv_);
        }
        //else if (select_module("azi_inc"))
        //{
            //return azi_inc(argc_, argv_)
        //}
        else
        {
            // TODO: Print: "unknown module"
            return 1;
        }
    }
    catch(std::exception& e)
    {
        // TODO: get info of exception and print it
        return 1;
    }
}

// extern "C"
}
