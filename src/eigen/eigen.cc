#include <string>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "aux/utils.hh"

#define Modules "poly_fit, eval_poly"

using namespace std;
using namespace utils;
using namespace Eigen;

static int fit_poly(int argc, char **argv)
{
    argparse ap(argc, argv, 
    "\n Usage: eigen fit_poly <coords> <deg> <is_centered> <fit_file>\
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
        return EARG;
    
    infile incoords;
    outfile fit_file;
    
    if (open(incoords, argv[2]) or open(fit_file, argv[5]))
        return EIO;
    
    double t_mean = 0.0,  // mean value of times
           t, x, y, z,    // temp storage variables
           x_mean = 0.0,  // x, y, z mean values
           y_mean = 0.0,
           z_mean = 0.0,
           t_min, t_max, res_tmp;
    
    
    vector<orbit_rec> orbits;
    uint ndata = 0;

    if (is_centered) {
        while(fscan(incoords, "%lf %lf %lf %lf\n", &t, &x, &y, &z) > 0) {
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
        
        while(fscan(incoords, "%lf %lf %lf %lf\n",
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
    
    ut_check(fprint(fit_file, "centered: %u\n", is_centered) < 0);
    
    if (is_centered) {
        ut_check(
            fprint(fit_file, "t_mean: %lf\n", t_mean) < 0 or
            fprint(fit_file, "coords_mean: %lf %lf %lf\n",
                             x_mean, y_mean, z_mean) < 0);
    }
    
    ut_check(
        fprint(fit_file, "t_min: %lf\n", t_min) < 0 or
        fprint(fit_file, "t_max: %lf\n", t_max) < 0 or
        fprint(fit_file, "deg: %u\n", deg) < 0 or
        fprint(fit_file, "coeffs: ") < 0
    );
    
    FOR(ii, 0, 3)
        FOR(jj, 0, deg + 1)
            ut_check(fprint(fit_file, "%lf ", fit(ii, jj)) < 0);

    //fprintf(fit_file, "\nRMS of residuals (x, y, z) [m]: (%lf, %lf, %lf)\n",
    //                  residual[0], residual[1], residual[2]);
    
    ut_check(fprint(fit_file, "\n") < 0);


fail: {
    errorln("Saving of fitted polynom values to file \"%s\" failed!",
            argv[5]);
    perror("fprint()");
    return EIO;
}
    
    return OK;
}

#if 0

static int eval_orbit(int argc, char **argv)
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

int main(int argc, char **argv)
{
    if (main_check_narg(argc, Modules))
        return EARG;
    
    string module_name(argv[1]);

    if (module_name == "azi_inc") {
        printf("azi_inc\n");
        return 0;
    }
    
    else if (module_name == "fit_orbit") {
        printf("fit_orbit\n");
        return 0;
    }
    else {
        errorln("Unrecognized module: %s", argv[1]);
        errorln("Modules to choose from: %s.", Modules);
        return EARG;
    }
    
    return OK;
}
