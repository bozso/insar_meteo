#!/usr/bin/env julia

push!(LOAD_PATH, "/home/istvan/progs/insar_meteo/inmet")

using PolynomFit

function main()
    println(MinMax(1.0, 2.0))
    #fit_orbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res",
    #          "doris", "orbit.fit")
    #load_fit("orbit.fit", display=true)
end

main()
