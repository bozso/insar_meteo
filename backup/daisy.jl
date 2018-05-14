#!/usr/bin/env julia

using SatOrbit

function main()
    
    a = Array{Float64, 2}(5, 4)
    
    ccall((:testfun, "libinsar"), Void, (Ptr{Cdouble}, Cuint, Cuint), a, 5, 4)
    
    println(a); return
    
    orbit_file = joinpath(homedir(), "progs", "insar_meteo", "daisy_test_data",
                          "asc_master.res")
    
    fit_orbit(orbit_file, "doris")

end

main()
