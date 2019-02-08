__precompile__(true)

module PolynomFit

using InteractiveUtils

export poly_fit, PolyFit, Scale, MinMax

struct Scale{T<:Number}
    min::T
    scale::T
    
    Scale(m::Tuple{T,T}) where T<:Number = new{T}(m[1], m[2] - m[1])
end

function Base.show(io::IO, p::Scale{T}) where T<:Number
    print(io, "Scale{$T} Min: $(p.min); Scale: $(p.scale)")
end


struct PolyFit{T<:Number}
    coeffs::VecOrMat{T}
    deg::Int64
    scaled::Bool
    xs::Union{Scale{T}, Nothing}
    ys::Union{Vector{Scale{T}}, Nothing}

    """
    f(x) = (x - x_min) / (x_max - x_min)
    
    f(x_max) = 1.0
    f(x_min) = 0.0
    """
end


function Base.show(io::IO, p::PolyFit{T}) where T<:Number
    print("PolyFit{$T} ")
    if p.scaled
        print(io, "Fit degree: $(p.deg), Scaled: true, x-scale: $(p.xs)\n",
                  "y-scale: $(p.ys)\nFitted Coefficients: $(p.coeffs)")
    else
        print(io, "Fit degree: $(p.deg), Scaled: false\n",
                  "Fitted Coefficients: $(p.coeffs)")
    end
end


function poly_fit(x::Vector{T}, y::VecOrMat{T}, deg::Integer,
                  scaled::Bool=false, dim::Integer=1) where T<:Number
    if scaled
        # TODO: properly calculate scales
        n = ndims(y)
        
        if n == 1
            dim > 1 && error("")
            yy = y
        elseif n == 2
            if dim == 1
                yy = y
            elseif dim == 2
                yy = transpose(y)
            else
                error("dim <= 2")
            end
        else
            error("Max 2 dim.")
        end
        
        search = hcat(x, yy)
        
        minmax = extrema(search, dims=1)
        
        xs = Scale(minmax[1])
        ys = [Scale(m) for m in minmax[2:end]]
        
        # f(x) = (x - x_min) / (x_max - x_min)
        
        design = collect( ((xx - xs.min) / xs.scale)^p for xx in x, p = 0:deg)
        
        if n == 1
            yys = ys[1]
            for jj in eachindex(yy)
                yy[jj] = (yy[jj] - yys.min) / yys.scale
            end
        else
            rows, cols = size(yy)
            for jj in 1:cols, ii in 1:rows
                yy[ii,jj] = (yy[ii,jj] - ys[jj].min) / ys[jj].scale
            end
        end
    else
        yy, xs, ys = y, nothing, nothing
        design = collect(xx^p for xx in x, p in 0:deg)
    end
    return PolyFit{T}(design \ yy, deg, scaled, xs, ys)
end

# module
end
