__precompile__(true)

module SatOrbit

using DelimitedFiles
using PolynomFit:fit_poly, PolyFit
using Serialization:serialize, deserialize

export read_orbits, fit_orbit, load_fit, Cart, Ellip, dot_product, ell_cart


const R_earth = 6372000.0;

const WA = 6378137.0
const WB = 6356752.3142

# (WA * WA - WB* WB) / WA / WA
const E2 = 6.694380e-03

const deg2rad = 1.745329e-02
const rad2deg = 5.729578e+01



"""
    orb = read_orbits(path::AbstractString, preproc::AbstractString)
    
    Read the orbit data (t, x(t), y(t), z(t)) stored in annotation files
    generated with DORIS or GAMMA preprocessing.
    
    # Example
    ```julia-repl
    julia> orb = read_orbits("INSAR_20150201/master.res", "doris");
    ```
"""
function read_orbits(path::String, preproc::String)
    if !isfile(path)
        error("$path does not exist!")
    end
    
    lines = readlines(path)
    
    if preproc == "doris"
        data_num = [(ii, line) for (ii, line) in enumerate(lines)
                               if startswith(line, "NUMBER_OF_DATAPOINTS:")]
        
        if length(data_num) != 1
            error("More than one or none of the lines contain the number of 
                   datapoints.")
        end
        
        idx = data_num[1][1]
        data_num = parse(Int, split(data_num[1][2], ":")[2])
        
        return transpose(readdlm(IOBuffer(join(lines[idx + 1:idx + data_num], "\n"))))
    elseif preproc == "gamma"
        error("Gamma processor not yet implemented!")
    else
        error("Unrecognized preprocessor option $preproc.");
    end
end


function fit_orbit(path::String, preproc::String,
                   savepath::Union{String,Nothing}=nothing, deg=3::Int64,
                   scaled=true::Bool)
    
    orb = read_orbits(path, preproc)
    
    fit = fit_poly(orb[1,:], orb[2:end,:], deg, scaled, 2)
    
    if savepath != nothing
        open(savepath, "w") do f
            serialize(f, fit)
        end
    end
    
    return fit
end


function load_fit(fit_file::String)
    open(fit_file, "r") do f
        fit = deserialize(f)
    end
end


struct Cart{T<:Number}
    x::T
    y::T
    z::T
end

struct Ellip{T<:Number}
    lon::T
    lat::T
    h::T
end


@inline function norm(x, y, z)
    return sqrt(x * x + y * y + z * z);
end


#=
@inline function calc_pos(orb::PolyFit, time::Float64, pos::Coord)
    x::Float64, y::Float64, z::Float64 = 0.0, 0.0, 0.0
    
    if (is_centered)
        time -= orb.mean_t;
    
    if(n_poly == 2) {
        x = coeffs(0, 0) * time + coeffs(0, 1);
        y = coeffs(1, 0) * time + coeffs(1, 1);
        z = coeffs(2, 0) * time + coeffs(2, 1);
    }
    else {
        x = coeffs(0, 0)  * time;
        y = coeffs(1, 0)  * time;
        z = coeffs(2, 0)  * time;

        m_for1(ii, 1, n_poly - 1) {
            x = (x + coeffs(0, ii)) * time;
            y = (y + coeffs(1, ii)) * time;
            z = (z + coeffs(2, ii)) * time;
        }
        
        x += coeffs(0, n_poly - 1);
        y += coeffs(1, n_poly - 1);
        z += coeffs(2, n_poly - 1);
    }
    
    if (is_centered) {
        x += mean_coords[0];
        y += mean_coords[1];
        z += mean_coords[2];
    }
    
    pos.x = x; pos.y = y; pos.z = z;
} // calc_pos
=#


"""
Calculate dot product between satellite velocity vector and
and vector between ground position and satellite position.
"""
@inline function dot_product(orb::PolyFit, X::Float64, Y::Float64, Z::Float64,
                             time::Float64)
    x, y, z, vx, vy, vz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    npoly, deg, scaled, coeffs = orb.deg + 1, orb.deg, orb.scaled, orb.coeffs

    @show size(coeffs)
    
    @inbounds begin
    
    if deg == 1
        if scaled
            scales = orb.scales
            
            tt = (time - scales[1].min) / scales[1].scale

            x = (coeffs[1, 1] * tt + coeffs[2, 1]) * scales[2].scale + scales[2].min
            y = (coeffs[1, 2] * tt + coeffs[2, 2]) * scales[3].scale + scales[3].min
            z = (coeffs[1, 3] * tt + coeffs[2, 3]) * scales[4].scale + scales[4].min

            vx = coeffs[1, 1] * scales[2].scale;
            vy = coeffs[1, 2] * scales[3].scale;
            vz = coeffs[1, 3] * scales[4].scale;
        else
            x = coeffs[1, 1] * time + coeffs[2, 1]
            y = coeffs[1, 2] * time + coeffs[2, 2]
            z = coeffs[1, 3] * time + coeffs[2, 3]

            vx = coeffs[1, 1]
            vy = coeffs[1, 2]
            vz = coeffs[1, 3]
        end

    # deg != 1
    else
    
        if scaled
            x, y, z = coeffs[1, 1] * tt, coeffs[1, 2] * tt, coeffs[1, 3] * tt
            vx, vy, vz = coeffs[2, 1], coeffs[2, 2], coeffs[2, 3]

            for ii in 2:deg
                x += coeffs[ii, 1] * tt
                y += coeffs[ii, 2] * tt
                z += coeffs[ii, 3] * tt
            end

            for ii in 3:npoly
                vx += ii * coeffs[ii, 1] * tt^(ii-1)
                vy += ii * coeffs[ii, 2] * tt^(ii-1)
                vz += ii * coeffs[ii, 3] * tt^(ii-1)
            end

            x = (x + coeffs[npoly, 1]) * scales[2].scale + scales[2].min
            y = (y + coeffs[npoly, 2]) * scales[3].scale + scales[3].min
            z = (z + coeffs[npoly, 3]) * scales[4].scale + scales[4].min
            
            vx, vy, vz = vx * scales[2].scale, vy * scales[3].scale, vz * scales[4].scale

        # not scaled
        else
            x, y, z = coeffs[1, 1] * time, coeffs[1, 2] * time, coeffs[1, 3] * time
            vx, vy, vz = coeffs[2, 1], coeffs[2, 2], coeffs[2, 3]

            for ii in 2:deg
                x += coeffs[ii, 1] * time
                y += coeffs[ii, 2] * time
                z += coeffs[ii, 3] * time
            end

            for ii in 3:npoly
                vx += ii * coeffs[ii, 1] * time^(ii-1)
                vy += ii * coeffs[ii, 2] * time^(ii-1)
                vz += ii * coeffs[ii, 3] * time^(ii-1)
            end

            x = x + coeffs[npoly, 1]
            y = y + coeffs[npoly, 2]
            z = z + coeffs[npoly, 3]
        
        # not scaled
        end
    
    # deg != 1
    end
    
    # @inbounds
    end

    # satellite coordinates - surface coordinates
    dx, dy, dz = x - X, y - Y, z - Z
    
    # product of inverse norms
    inorm = (1.0 / norm(dx, dy, dz)) * (1.0 / norm(vx, vy, vz))
    
    return (vx * dx  + vy * dy  + vz * dz) * inorm

# dot_product
end

#=
"""Compute the sat position using closest approche."""
@inline function closest_appr(orb::PolyFit, X::Float64, Y::Float64, Z::FLoat64,
                              max_iter::Int64)
    
    # first, last and middle time, extending the time window by 5 seconds
    t_start = orb.scales[0].min - 5.0,
    t_stop  = orb.scales[0].min + orb.scales[0].scale + 5.0,
    t_middle = (t_start - t_stop) / 2.0;
    
    # dot products
    dot_start, dot_middle = 0.0, 1.0;

    # iteration counter
    itr = 0;
    
    dot_start = dot_product(orb, X, Y, Z, t_start);
    
    while (abs(dot_middle) > 1.0e-11 and itr < max_iter) {
        t_middle = (t_start + t_stop) / 2.0;

        dot_middle = dot_product(orb, X, Y, Z, t_middle);
        
        // change start for middle
        if ((dot_start * dot_middle) > 0.0) {
            t_start = t_middle;
            dot_start = dot_middle;
        }
        // change  end  for middle
        else
            t_stop = t_middle;

        itr++;
    } // while
    
    // calculate satellite position at middle time
    calc_pos(orb, t_middle, sat_pos);
} // closest_appr
=#

function ell_cart(e::Ellip)
    slat, clat, clon, slon, h = sin(e.lat), cos(e.lat), cos(e.lon), sin(e.lon), e.h
    
    n = WA / sqrt(1.0 - E2 * slat * slat)

    x = (              n + h) * clat * clon
    y = (              n + h) * clat * slon
    z = ( (1.0 - E2) * n + h) * slat

    return Cart(x, y, z)

# ell_cart
end


function cart_ell(c::Cart)
    x, y, z = c.x, c.z, c.z
    
    n = (WA * WA - WB * WB);
    p = sqrt(x * x + y * y);

    o = atan(WA / p / WB * z);
    
    so = sin(o)
    co = cos(o)
    
    o = atan( (z + n / WB * so * so * so) / (p - n / WA * co * co * co) )
    
    so = sin(o)
    co = cos(o);
    
    n = WA * WA / sqrt(WA * co * co * WA + WB * so * so * WB)
    lat = o
    
    o = atan(y/x)
    
    if x < 0.0
        o += pi
    end
    
    lon = o
    h = p / co - n
    
    return Ellip(lon, lat, h)

# cart_ell
end

#=
static inline void _azi_inc(fit_poly const& orb, cdouble X, cdouble Y,
                            cdouble Z, cdouble lon, cdouble lat,
                            size_t max_iter, double& azi, double& inc)
{
    double xf, yf, zf, xl, yl, zl, t0;
    cart sat;
    
    // satellite closest approache cooridantes
    closest_appr(orb, X, Y, Z, max_iter, sat);
    
    xf = sat.x - X;
    yf = sat.y - Y;
    zf = sat.z - Z;
    
    // estiamtion of azimuth and inclination
    xl = - sin(lat) * cos(lon) * xf
         - sin(lat) * sin(lon) * yf + cos(lat) * zf ;
    
    yl = - sin(lon) * xf + cos(lon) * yf;
    
    zl = + cos(lat) * cos(lon) * xf
         + cos(lat) * sin(lon) * yf + sin(lat) * zf ;
    
    t0 = norm(xl, yl, zl);
    
    inc = acos(zl / t0) * rad2deg;
    
    if(xl == 0.0) xl = 0.000000001;
    
    double temp_azi = atan(abs(yl / xl));
    
    if( (xl < 0.0) && (yl > 0.0) ) temp_azi = consts::pi - temp_azi;
    if( (xl < 0.0) && (yl < 0.0) ) temp_azi = consts::pi + temp_azi;
    if( (xl > 0.0) && (yl < 0.0) ) temp_azi = 2.0 * consts::pi - temp_azi;
    
    temp_azi *= rad2deg;
    
    if(temp_azi > 180.0)
        temp_azi -= 180.0;
    else
        temp_azi += 180.0;
    
    azi = temp_azi;
}
// _azi_inc


void calc_azi_inc(fit_poly const& orb, View<double> const& coords,
                  View<double>& azi_inc, size_t const max_iter,
                  uint const is_lonlat)
{
    double X, Y, Z, lon, lat, h;
    X = Y = Z = lon = lat = h = 0.0;
    
    size_t nrows = coords.shape(0);
    
    // coords contains lon, lat, h
    if (is_lonlat) {
        m_for(ii, nrows) {
            lon = coords(ii, 0) * deg2rad;
            lat = coords(ii, 1) * deg2rad;
            h   = coords(ii, 2);
            
            // calulate surface WGS-84 Cartesian coordinates
            ell_cart(lon, lat, h, X, Y, Z);
            
            _azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                    azi_inc(ii, 0), azi_inc(ii, 1));
            
        } // for
    }
    // coords contains X, Y, Z
    else {
        m_for(ii, nrows) {
            X = coords(ii, 0);
            Y = coords(ii, 1);
            Z = coords(ii, 2);
            
            // calulate surface WGS-84 geodetic coordinates
            cart_ell(X, Y, Z, lon, lat, h);
        
            _azi_inc(orb, X, Y, Z, lon, lat, max_iter,
                    azi_inc(ii, 0), azi_inc(ii, 1));
        } // for
    } // if
}
=#


#function azi_inc(fit_file::AbstractString, coords::Array{T, 2},
#                 is_lonlat=true::Bool, max_iter=1000::Int) where T<:Real
    
#    t_start, t_stop, t_mean, coeffs, mean_coords, is_centered, deg = \
#    load_fit(fit_file)
    
#    inarg = (Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble},
#             Ptr{Cdouble}, Ptr{Cdouble}, Cuint, Cuint, Cuint, Cuint, Cuint)
    
#    ndata = size(coords, 1)
    
#    ret = Array{Float64, 2}(ndata, 2)
    
#    ccall((:azi_inc, "libinsar"), (Void,), inarg, t_start, t_stop, t_mean,
#                                  coeffs, coords, mean_coords, ret, ndata,
#                                  is_centered, deg, max_iter, is_lonlat)
#    ret
#end

end
