#!/usr/bin/env julia

push!(LOAD_PATH, "/home/istvan/progs/insar_meteo/inmet")

using PolynomFit

function main()
    #fit_orbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res",
    #          "doris", "orbit.fit")
    #load_fit("orbit.fit", display=true)
end

main()
