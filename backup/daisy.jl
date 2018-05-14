#!/usr/bin/env julia

using SatOrbit

function main()
    ccall((:azi_inc, "libaux"), Void, (Cdouble, Cdouble, Cdouble), 0.0, 1.0, 2.0)
    return
    
    orbit_file = joinpath(homedir(), "progs", "insar_meteo", "daisy_test_data",
                          "asc_master.res")
    
    fit_orbit(orbit_file, "doris")

end

main()
