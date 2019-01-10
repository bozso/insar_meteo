#include <stdio.h>
#include <vector>

using std::vector;

#include "array.h"
#include "satorbit.h"

extern "C" {

int test(arptr _arr)
{
    View<int64_t> const arr{_arr};
    
    printf("%lu\n", arr.ndim);
    
    for(size_t ii = 0; ii < arr.shape[0]; ++ii)
    {
        for(size_t jj = 0; jj < arr.shape[1]; ++jj)
            printf("%ld ", arr(ii, jj));
        printf("\n");
    }
    
    printf("\n");

    return 0;
}


int ell_to_merc(arptr plon, arptr plat, arptr pxy, double a, double e,
                double lon0, size_t isdeg, size_t fast)
{
    // check size, datatype
    try
    {
        if (plon->check_type(np_float64) or plat->check_type(np_float64) or
            pxy->check_type(np_float64))
            return 1;
        
        View<double> lon{plon}, lat{plat}, xy{pxy};
        size_t rows = lon.shape[0];
        
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
    catch(...)
    {
        // handle errors
        return 1;
    }
}
// ell_to_merc


int azi_inc(arptr pmean, arptr pcoeffs, arptr pcoords, arptr pazi_inc,
            double mean_t, double start_t, double stop_t, size_t is_centered,
            size_t deg, size_t islonlat, size_t max_iter)
{
    try 
    {
        View<double> coeffs{pcoeffs}, coords{pcoords}, azi_inc{pazi_inc};
        
        // Set up orbit polynomial structure
        fit_poly orb{mean_t, start_t, stop_t, (double*) pmean->get_data(),
                     coeffs, is_centered, deg};
        
        calc_azi_inc(orb, coords, azi_inc, max_iter, islonlat);
        
        return 0;
    }
    catch(...)
    {
        // handle errors
        return 1;
    }
}
// azi_inc


int asc_dsc_select(arptr parr1, arptr parr2, arptr pidx, double max_sep,
                   size_t* nfound)
{
    try
    {
        View<double> arr1{parr1}, arr2{parr2};
        View<char> idx{pidx};
        
        max_sep /=  R_earth;
        max_sep = (max_sep * rad2deg) * (max_sep * rad2deg);
        
        *nfound = 0;
        size_t n1 = arr1.shape[0], n2 = arr2.shape[0];
        
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
    }
    catch(...)
    {
        // handle errors
        return 1;
    }
    return 0;

} // asc_dsc_select

int dominant(arptr pasc, arptr pdsc, arptr clustered, double max_sep)
{
    try
    {
        View<double> asc{pasc}, dsc{pdsc};
        vector<bool> asc_selected, dsc_selected;
        vector<double> clustered;
        
        asc_selected.reserve(asc.shape[0]);
        dsc_selected.reserve(dsc.shape[0]);
        
        
    }
    catch(...)
    {
        // handle errors
        return 1;
    }
    return 0;
}
// dominant


}
// extern "C"
