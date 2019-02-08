#!/usr/bin/env julia

module Test
    include("../inmet/PolynomFit.jl")
    
    using InteractiveUtils
    using .PolynomFit
    
    precompile(poly_fit, (Vector{Float64}, Matrix{Float64}, Int64, Bool))
    @show x, y = [1.0; 2.0; 3.0;], [4.0 5.0; 5.0 6.0; 6.0 7.0]
    
    fit = poly_fit(x, y, 1, true)

    @show yy = fit(x)
    
    #fit_orbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res",
    #          "doris", "orbit.fit")
    #load_fit("orbit.fit", display=true)
end
