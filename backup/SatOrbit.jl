#!/usr/bin/env julia

module SatOrbit

export read_orbits, fit_orbit

"""
    orb = read_orbits(path::AbstractString, preproc::AbstractString)
    
    Read the orbit data (t, x(t), y(t), z(t)) stored in annotation files
    generated with DORIS or GAMMA preprocessing.
    
    # Example
    ```julia-repl
    julia> orb = read_orbits("INSAR_20150201/master.res", "doris");
    ```
"""
function read_orbits(path::AbstractString, preproc::AbstractString)
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
        
        return readdlm(IOBuffer(join(lines[idx + 1:idx + data_num], "\n"))).'
    elseif preproc == "gamma"
    
    
    else
        error("Unrecognized preprocessor option $preproc.");
    end
end

function fit_orbit(path::AbstractString, preproc::AbstractString, deg=3::Int,
                   centered=true::Bool)
    
    orb = read_orbits(path, preproc)
    
    ndata = size(orb, 2)
    
    if centered
        t_mean = mean(orb[1,:])
        time = orb[1,:] - t_mean
        
        coords_mean = mean(orb[2:4,:], 2)
        coords = orb[2:4,:] .- coords_mean
        cent = "centered: 1"
    else
        time = orb[1,:]
        coords = orb[2:4,:]
        cent = "centered: 0"
    end
    
    design = Array{Float64, 2}(ndata, deg + 1)
    
    design[:,end] = 1.0
    design[:,end-1] = time

    design[:,1:end-2] = collect(v^p for v in time, p in deg:-1:2)
    
    @show size(design), size(coords.')
    
    @show fit = design \ coords.'
    
    return
end

function load_fit(fit_file::AbstractString)


end

function azi_inc(fit_file::AbstractString, coords::Array{T, 2},
                 is_lonlat=true::Bool, max_iter=1000::Int) where T<:Real
    
    t_start, t_stop, t_mean, coeffs, mean_coords, is_centered, deg = \
    load_fit(fit_file)
    
    inarg = (Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Cuint, Cuint, Cuint, Cuint, Cuint)
    
    ndata = size(coords, 1)
    
    ret = Array{Float64, 2}(ndata, 2)
    
    ccall((:azi_inc, "libinsar"), Void, inarg, t_start, t_stop, t_mean,
                                  coeffs, coords, mean_coords, ret, ndata,
                                  is_centered, deg, max_iter, is_lonlat)
    ret
end

end
