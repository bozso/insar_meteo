/* Copyright (C) 2018  István Bozsó
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <vector>

#include "utils.hpp"
#include "main_functions.hpp"

#define min_arg 2

using namespace std;

/************************
 * Auxilliary functions *
 * **********************/

struct argparse {
    int argc;
    char **argv;
    const char * usage;
    
    argparse(int _argc, char **_argv, const char * _usage):
        argc(_argc), argv(_argv), usage(_usage) {};
    
    void print_usage() const
    {
        error("Usage: \n%s\n", usage);
    }
};

static bool check_narg(const argparse& ap, int req_arg)
{
    if (ap.argc != (req_arg + min_arg)) {
        errorln("\n Required number of arguments is %d, current number of "
                "arguments: %d!\n", req_arg, ap.argc - min_arg);
        ap.print_usage();
        return true;
    }
    return false;
};

template<typename T>
static bool get_arg(const argparse& ap, const uint idx, const char * fmt,
                    T& target)
{
    if (sscanf(ap.argv[1 + idx], fmt, &target) != 1) {
        errorln("Invalid argument: %s", ap.argv[1 + idx]);
        ap.print_usage();
        return true;
    }
    return false;
}

/***********************************************
 * Main functions - calleble from command line *
 ***********************************************/

int fit_orbit(int argc, char **argv)
{
    argparse ap(argc, argv, 
    "\n Usage: inmet fit_orbit <coords> <deg> <is_centered> <fit_file>\
     \n \
     \n coords      - (ascii, in) file with (t,x,y,z) coordinates\
     \n deg         - degree of fitted polynom\
     \n is_centered - 1 = subtract mean time and coordinates from time points and \
     \n               coordinates, 0 = no centering\
     \n fit_file    - (ascii, out) contains fitted orbit polynom parameters\
     \n\n");
    
    uint deg = 0, is_centered = 0;
    
    if (check_narg(ap, 4) or
        get_arg(ap, 2, "%u", deg) or get_arg(ap, 2, "%u", is_centered))
        return 1;
    
    File incoords, fit_file;
    
    if (open(incoords, argv[2], "r") or open(fit_file, argv[5], "w"))
        return 1;
    
    double t_mean = 0.0,  // mean value of times
           t, x, y, z,    // temp storage variables
           x_mean = 0.0,  // x, y, z mean values
           y_mean = 0.0,
           z_mean = 0.0,
           t_min, t_max, res_tmp;
    
    
    vector<orbit_rec> orbits;
    uint ndata = 0;

    if (is_centered) {
        while(read(incoords, "%lf %lf %lf %lf\n", &t, &x, &y, &z) > 0) {
            t_mean += t;
            x_mean += x;
            y_mean += y;
            z_mean += z;
            
            orbit_rec tmp = {t, x, y, z};
            
            orbits.push_back(tmp);
        }
        ndata = orbits.size();
        
        // calculate means
        t_mean /= double(ndata);
    
        x_mean /= double(ndata);
        y_mean /= double(ndata);
        z_mean /= double(ndata);
    }
    else {
        orbit_rec tmp = {0};
        
        while(read(incoords, "%lf %lf %lf %lf\n",
                     &tmp.t, &tmp.x, &tmp.y, &tmp.z) > 0) {
            orbits.push_back(tmp);
        }
        
        ndata = orbits.size();
    }
    
    t_min = orbits[0].t;
    t_max = orbits[0].t;
    
    FOR(ii, 1, ndata) {
        t = orbits[ii].t;
        
        if (t < t_min)
            t_min = t;

        if (t > t_max)
            t_max = t;

    }

    mat obs(ndata, 3);
    mat fit(3, deg + 1);

    mat design(ndata, deg + 1);
    
    FOR(ii, 0, ndata) {
        // fill up matrix that contains coordinate values
        obs(ii, 0) = orbits[ii].x - x_mean;
        obs(ii, 1) = orbits[ii].y - y_mean;
        obs(ii, 2) = orbits[ii].z - z_mean;
        
        t = orbits[ii].t - t_mean;
        
        // fill up design matrix
        
        // first column is ones
        design(ii, 0) = 1.0;
        
        // second column is t values
        design(ii, 1) = t;
        
        // further columns contain the power of t values
        FOR(jj, 2, deg + 1)
            design(ii, jj) = design(ii, jj - 1) * t;
    }

    
    write(fit_file, "centered: %u\n", is_centered);
    
    if (is_centered) {
        write(fit_file, "t_mean: %lf\n", t_mean);
        write(fit_file, "coords_mean: %lf %lf %lf\n",
                                        x_mean, y_mean, z_mean);
    }
    
    write(fit_file, "t_min: %lf\n", t_min);
    write(fit_file, "t_max: %lf\n", t_max);
    write(fit_file, "deg: %u\n", deg);
    write(fit_file, "coeffs: ");
    
    FOR(ii, 0, 3)
        FOR(jj, 0, deg + 1)
            fprintf(fit_file, "%lf ", fit(ii, jj));

    //fprintf(fit_file, "\nRMS of residuals (x, y, z) [m]: (%lf, %lf, %lf)\n",
    //                  residual[0], residual[1], residual[2]);
    
    write(fit_file, "\n");
    
    return 0;
}

#if 0

int eval_orbit(int argc, char **argv)
{
    int error = err_succes;

    checkarg(argc, 4,
    "\n Usage: inmet eval_orbit [fit_file] [steps] [multiply] [outfile]\
     \n \
     \n fit_file    - (ascii, in) contains fitted orbit polynom parameters\
     \n nstep       - evaluate x, y, z coordinates at nstep number of steps\
     \n               between the range of t_min and t_max\
     \n multiply    - calculated coordinate values will be multiplied by this number\
     \n outfile     - (ascii, out) coordinates and time values will be written \
     \n               to this file\
     \n\n");

    FILE *outfile;
    orbit_fit orb;
    double t_min, t_mean, dstep, t, nstep, mult =  atof(argv[4]);
    cart pos;
    
    if ((error = read_fit(argv[2], &orb))) {
        errorln("Could not read orbit fit file %s. Exiting!", argv[2]);
        return error;
    }
    
    t_min = orb.t_min;
    nstep = atof(argv[3]);
    
    dstep = (orb.t_max - t_min) / nstep;
    
    aux_open(outfile, argv[5], "w");
    
    if (orb.centered)
        t_mean = orb.t_mean;
    else
        t_mean = 0.0;

    FOR(ii, 0, ((uint) nstep) + 1) {
        t = t_min - t_mean + ii * dstep;
        calc_pos(&orb, t, &pos);
        fprintf(outfile, "%lf %lf %lf %lf\n", t + t_mean, pos.x * mult,
                                                          pos.y * mult,
                                                          pos.z * mult);
    }

    fclose(outfile);
    return error;

fail:
    aux_close(outfile);
    return error;
}

int azi_inc(int argc, char **argv)
{
    int error = err_succes;
    
    checkarg(argc, 5,
    "\n Usage: inmet azi_inc [fit_file] [coords] [mode] [max_iter] [outfile]\
     \n \
     \n fit_file - (ascii, in) contains fitted orbit polynom parameters\
     \n coords   - (binary, in) inputfile with coordinates\
     \n mode     - xyz for WGS-84 coordinates, llh for WGS-84 lon., lat., height\
     \n max_iter - maximum number of iterations when calculating closest approache\
     \n outfile  - (binary, out) azi, inc pairs will be printed to this file\
     \n\n");

    FILE *infile, *outfile;
    uint is_lonlat, max_iter = atoi(argv[5]);
    double coords[3];
    orbit_fit orb;

    // topocentric parameters in PS local system
    double X, Y, Z,
           lon, lat, h,
           azi, inc;
    
    if ((error = read_fit(argv[2], &orb))) {
        errorln("Could not read orbit fit file %s. Exiting!", argv[2]);
        return error;
    }
    
    aux_open(infile, argv[3], "rb");
    aux_open(outfile, argv[6], "wb");
    
    // infile contains lon, lat, h
    if (str_isequal(argv[4], "llh")) {
        while (fread(coords, sizeof(double), 3, infile) > 0) {
            lon = coords[0] * DEG2RAD;
            lat = coords[1] * DEG2RAD;
            h   = coords[2];
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &X, &Y, &Z);
            
            calc_azi_inc(&orb, X, Y, Z, lon, lat, max_iter, &azi, &inc);

            fwrite(&azi, sizeof(double), 1, outfile);
            fwrite(&inc, sizeof(double), 1, outfile);
        } // end while
    }
    // infile contains X, Y, Z
    else if (str_isequal(argv[4], "xyz")) {
        while (fread(coords, sizeof(double), 3, infile) > 0) {
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, &coords[0], &coords[1], &coords[2]);
            
            calc_azi_inc(&orb, coords[0], coords[1], coords[2],
                         lon, lat, max_iter, &azi, &inc);
            
            fwrite(&azi, sizeof(double), 1, outfile);
            fwrite(&inc, sizeof(double), 1, outfile);
        } // end while
    } // end else if
    else {
        errorln("Third argument should be either llh or xyz not %s!",
                argv[4]);
        error = err_arg;
        goto fail;
    }

    fclose(infile);
    fclose(outfile);
    return error;

fail:
    aux_close(infile);
    aux_close(outfile);
    return error;
}

#endif

#if 0

#define SIZE 2500

int test_matrix1(void)
{
    matrix * mtx1, *mtx2, *mtx3;
    mtx_double(mtx1, SIZE, SIZE);
    mtx_double(mtx2, SIZE, SIZE);
    mtx_double(mtx3, SIZE, SIZE);
    
    FOR(ii, 0, SIZE) {
        FOR(jj, 0, SIZE) {
            dmtx(mtx1, ii, jj) = (double) ii + jj;
            dmtx(mtx2, ii, jj) = (double) jj + ii;
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, SIZE, SIZE, SIZE, 1.0, (double *)mtx1->data, SIZE, (double *)mtx2->data, SIZE, 0.0, (double *)mtx3->data, SIZE);

    FOR(ii, 0, 10)
        printf("%lf ", dmtx(mtx3, 0, ii));
    
    printf("\n");

    
    mtx_free(mtx1);
    mtx_free(mtx2);
    mtx_free(mtx3);
    return 0;
fail:
    mtx_safe_free(mtx1);
    mtx_safe_free(mtx2);
    mtx_safe_free(mtx3);
    return 1;
}

int test_matrix2(void)
{
    
    gsl_matrix * mtx1 = gsl_matrix_alloc(SIZE, SIZE);
    gsl_matrix * mtx2 = gsl_matrix_alloc(SIZE, SIZE);
    gsl_matrix * mtx3 = gsl_matrix_alloc(SIZE, SIZE);
    
    FOR(ii, 0, SIZE) {
        FOR(jj, 0, SIZE) {
            Mset(mtx1, ii, jj, (double) ii + jj);
            Mset(mtx2, ii, jj, (double) jj + ii);
        }
    }
    
    gsl_blas_dgemm(CblasTrans, CblasTrans, 1.0, mtx1, mtx2, 0.0, mtx3);
    
    FOR(ii, 0, 10)
        printf("%lf ", Mget(mtx3, 0, ii));
    
    printf("\n");
    
    gsl_matrix_free(mtx1);
    gsl_matrix_free(mtx2);
    gsl_matrix_free(mtx3);
    return 0;
}

#endif
