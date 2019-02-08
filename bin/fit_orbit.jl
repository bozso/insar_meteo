#!/usr/bin/env julia

module Test
    include("../inmet/PolynomFit.jl")
    
    using InteractiveUtils
    using .PolynomFit
    
    fit = poly_fit([1.0; 2.0; 3.0;], [4.0 5.0; 5.0 6.0; 6.0 7.0], 1, true)
    
    @show fit.coeffs
    
    for a in fit.coeffs
        println(a)
    end

    #fit_orbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res",
    #          "doris", "orbit.fit")
    #load_fit("orbit.fit", display=true)
end
