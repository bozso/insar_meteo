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


#include "utils.hh"
#include "math_aux.hh"

matrix mtx_init(const size_t rows, const size_t cols, const datatype dtype)
{
    matrix tmp;
    
    switch(dtype) {
        case dt_cx128:
            tmp.mtx.cx128 = gsl_matrix_complex_long_double_alloc(rows, cols);
            break;
        case dt_cx64:
            tmp.mtx.cx64 = gsl_matrix_complex_alloc(rows, cols);
            break;
        case dt_cx32:
            tmp.mtx.cx32 = gsl_matrix_complex_float_alloc(rows, cols);
            break;

        case dt_fl128:
            tmp.mtx.fl128 = gsl_matrix_long_double_alloc(rows, cols);
            break;
        case dt_fl64:
            tmp.mtx.fl64 = gsl_matrix_alloc(rows, cols);
            break;
        case dt_fl32:
            tmp.mtx.fl32 = gsl_matrix_float_alloc(rows, cols);
            break;
        
        default:
            errorln("Unknown matrix type!");
    }
    return tmp;
}

matrix::~matrix()
{
    switch(dtype) {
        case dt_cx128:
            gsl_matrix_complex_long_double_free(mtx.cx128);
            break;
        case dt_cx64:
            gsl_matrix_complex_free(mtx.cx64);
            break;
        case dt_cx32:
            gsl_matrix_complex_float_free(mtx.cx32);
            break;

        case dt_fl128:
            gsl_matrix_long_double_free(mtx.fl128);
            break;
        case dt_fl64:
            gsl_matrix_free(mtx.fl64);
            break;
        case dt_fl32:
            gsl_matrix_float_free(mtx.fl32);
            break;
    }
}

static datatype str2dt(const char * str);
static const char * dt2str(datatype dtype);
static store_type str2store(const char * str);
static const char * store2str(store_type storage);


int parse_parfile(const char * parfile_path, size_t& rows, size_t& cols,
                  store_type& storage, datatype& dtype)
{
    File parfile;
    
    if (open(parfile, parfile_path, "r"))
        return EIO;
    
    const char *dtype_str = NULL, *storage_str = NULL;
    
    if (
    read(parfile, "rows: %u\n", &rows) < 0 or
    read(parfile, "cols: %u\n", &cols) < 0 or
    read(parfile, "storage type: %s\n", &storage_str) < 0 or
    read(parfile, "dtype: %s\n", &dtype_str) < 0
    ) {
        perrorln("parse_parfile", "Could not read parameter file properly!");
        return EIO;
    }
    close(parfile);
    
    dtype = str2dt(dtype_str);
    storage = str2store(storage_str);
    
    if (dtype == dt_unk) {
        errorln("Unknown matrix type \"%s\"", dtype_str);
        return EIO;
    }

}


int write_parfile(const char * parfile_path, const size_t rows,
                  const size_t cols, const store_type storage,
                  const datatype dtype)
{
    File parfile;
    
    if (open(parfile, parfile_path, "w"))
        return EIO;
    
    if (
    write(parfile, "rows: %u\n", rows) < 0 or
    write(parfile, "cols: %u\n", cols) < 0 or
    write(parfile, "storage type: %s\n", store2str(storage)) < 0 or
    write(parfile, "dtype: %s\n", dt2str(dtype)) < 0) {
        perrorln("write_parfile", "Could not write parameter file properly!");
        return EIO;
    }
    close(parfile);
    return OK;
}


int mtx_read(matrix& mat, const char * datafile_path, const char * parfile_path)
{
    size_t rows, cols;
    store_type storage;
    datatype dtype;
    
    parse_parfile(parfile_path, rows, cols, storage, dtype);
    
    mat = mtx_init(rows, cols, dtype);
    
    File datafile;
    
    if (open(datafile, datafile_path, "r"))
        return EIO;

    //mtx_fread(mat, 

    return OK;
}

bool read_fit(fit_poly& fit, const char * filename)
{
    //infile 
    
}


#if 0

int poly_fit(int argc, char **argv)
{
    argparse ap = {argc, argv, 
    "\n Usage: inmet fit_poly <coords> <deg> <is_centered> <fit_file>\
     \n \
     \n coords      - (ascii, in) file with (t,x,y,z) coordinates\
     \n deg         - degree of fitted polynom\
     \n is_centered - 1 = subtract mean time and coordinates from time points and \
     \n               coordinates, 0 = no centering\
     \n fit_file    - (ascii, out) contains fitted orbit polynom parameters\
     \n\n"};
    
    uint deg = 0, is_centered = 0;
    
    if (check_narg(ap, 4) or
        get_arg(ap, 2, "%u", &deg) or get_arg(ap, 2, "%u", &is_centered))
        return EARG;
    
    uint ndata = 0;

    if (is_centered) {
        while(scan(incoords, "%lf %lf %lf %lf\n", &t, &x, &y, &z) > 0) {
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
        
        while(scan(incoords, "%lf %lf %lf %lf\n",
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

    MatrixXd obs(ndata, 3);
    MatrixXd fit(3, deg + 1);

    MatrixXd design(ndata, deg + 1);
    
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
    
    fit = (design * design.transpose()).llt().solve(design.transpose() * obs);
    
    ut_check(print(fit_file, "centered: %u\n", is_centered) < 0);
    
    if (is_centered) {
        ut_check(
            fprintf(fit_file, "x_mean: %lf\n", t_mean) < 0 or
            fprintf(fit_file, "dependents_mean: %lf %lf %lf\n",
                             x_mean, y_mean, z_mean) < 0);
    }
    
    ut_check(
        fprintf(fit_file, "x_min: %lf\n", t_min) < 0 or
        fprintf(fit_file, "x_max: %lf\n", t_max) < 0 or
        fprintf(fit_file, "deg: %u\n", deg) < 0 or
        fprintf(fit_file, "coeffs: ") < 0
    );
    
    FOR(ii, 0, 3)
        FOR(jj, 0, deg + 1)
            ut_check(print(fit_file, "%lf ", fit(ii, jj)) < 0);

    //fprintf(fit_file, "\nRMS of residuals (x, y, z) [m]: (%lf, %lf, %lf)\n",
    //                  residual[0], residual[1], residual[2]);
    
    ut_check(print(fit_file, "\n") < 0);


fail: {
    errorln("Saving of fitted polynom values to file \"%s\" failed!",
            argv[5]);
    perror("print()");
    return EIO;
}
    
    return OK;
}


int poly_eval(int argc, char **argv)
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

#endif

static datatype str2dt(const char * str)
{
    if (not strcmp(str, "complex long double"))
        return dt_cx128;
    else if (not strcmp(str, "complex double"))
        return dt_cx64;
    else if (not strcmp(str, "complex float"))
        return dt_cx32;
    else if (not strcmp(str, "long double"))
        return dt_fl128;
    else if (not strcmp(str, "double"))
        return dt_fl64;
    else if (not strcmp(str, "float"))
        return dt_fl32;
    else {
        errorln("Unknown matrix type \"%s\"", str);
        return dt_unk;
    }
}

static const char * dt2str(datatype dtype)
{
    switch(dtype) {
        case dt_cx128:
            return "complex long double";
        case dt_cx64:
            return "complex double";
        case dt_cx32:
            return "complex float";

        case dt_fl128:
            return "long double";
        case dt_fl64:
            return "double";
        case dt_fl32:
            return "float";
        
        case dt_unk:
            errorln("Unknown matrix type!");
            return "unknown";
        default:
            errorln("Unknown matrix type!");
            return "unknown";
    }

}

static store_type str2store(const char * str)
{
    if (str_equal(str, "ascii"))
        return ascii;
    else if (str_equal(str, "binary"))
        return binary;
    else {
        errorln("Unknown storage type \"%\"!", str);
        return unk;
    }
}

static const char * store2str(store_type storage)
{
    switch(storage) {
        case ascii:
            return "ascii";
        case binary:
            return "binary";
        case unk:
            errorln("Unknown storage type!");
            return "unknown";
        default:
            errorln("Unknown storage type!");
            return "unknown";
    }
}
