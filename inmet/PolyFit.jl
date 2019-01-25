__precompile__(true)

module PolyFit

using Statistics:mean, mean!

export poly_fit, PolyFit

struct PolyC{T} where T <: Number
    coeffs::AbstractVecOrMath{T}
    xm::T
    ym::AbstractVector{T}
end

struct PolyNC{T} where T <: Number
    coeffs::AbstractVecOrMath{T}
end

struct PolyFit{T} where T <: Number
    poly::Union{PolyC{T}, PolyNC{T}}
    deg::Integer
    centered::Bool
end


function Base.show(io::IO, p::PolyFit)
    centered = p.centered
    if centered
        print(io, "Fit degree: $p.deg, Centered:$centered, x-mean: $p.xm\n",
                  "y-mean: $p.ym\nFitted Coefficients: $p.coeffs")
    else
        print(io, "Fit degree: $p.deg, Centered:$centered\n",
                  "Fitted Coefficients: $p.coeffs")
    end
end


function poly_fit(x::AbstrectVector, y::AbstractVecOrMat, deg::Integer,
                  centered::Bool=false)
    elt = eltype(y)
    if centered
        # TODO: properly calculate means
        xm, ym = mean(x), Vector{elt}(undef, size(y, dim))
        mean!(ym, y)
        return PolyFit{elt}(collect(v ^ p for v in x, p in 0:deg) \ y,
                            deg, centered, xm, ym)
    else
        return PolyFit{elt}(collect(v ^ p for v in x, p in 0:deg) \ y,
                            deg, centered, none, none)
    end
end

# module
end
