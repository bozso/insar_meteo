#!/usr/bin/env julia

module Test
    push!(LOAD_PATH, "../inmet/")
    
    include("../inmet/SatOrbit.jl")
    
    using InteractiveUtils
    using .SatOrbit
        
    fit = fit_orbit("/home/istvan/progs/insar_meteo/daisy_test_data/asc_master.res",
                    "doris", nothing)
    
    a = Ellip(25.5515, 46.33215, 19.3452)
    
    dot_product(fit, ell_cart(a))
    
    @show load_fit("orbit.fit")
    
    #load_fit("orbit.fit", display=true)
end
