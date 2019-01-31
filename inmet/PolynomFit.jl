__precompile__(true)

module PolynomFit

export poly_fit, PolyFit, Scale, MinMax

struct MinMax{T<:Number}
    min::T
    max::T
end

function Base.show(io::IO, p::MinMax{T}) where T<:Number
    print(io, "Min: $p.min; Max: $p.max")
end


struct Scale{T<:Number}
    min::T
    scale::T
    
    Scale(m::MinMax{T}) where T<:Number = new{T}(m.min, m.max - m.min)
end

function Base.show(io::IO, p::Scale)
    print(io, "Min: $p.min; Scale: $p.scale")
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

function Base.show(io::IO, p::PolyFit)
    scaled = p.scaled
    if centered
        print(io, "Fit degree: $p.deg, Centered:$centered, x-scale: $p.xs\n",
                  "y-scale: $p.ys\nFitted Coefficients: $p.coeffs")
    else
        print(io, "Fit degree: $p.deg, Centered:$centered\n",
                  "Fitted Coefficients: $p.coeffs")
    end
end


function poly_fit(x::Vector{T}, y::VecOrMat{T}, deg::Integer,
                  scaled::Bool=false) where T<:Number
    elt = eltype(y)
    
    if scaled
        # TODO: properly calculate scales
        yrows, ycols = size(y)
        
        if dim == 1
            ysize1, ysize2 = yrows, ycols
        else
            ysize1, ysize2 = ycols, yrows
        end
        
        if length(x) != ysize1
            error("")
        end
        
        xm::MinMax{elt} = [x[1], x[1]]
        ym::Vector{MinMax{elt}}(undef, ysize2)
        
        xt = 0::elt
        
        @inbounds begin
        for ii = 2:ysize
            xt = x[ii]
            
            if xt < xs.min
                x.min = xt
            end
            
            if xt > xs.max
                xs.max = xt
            end
            
            yt = selectdim(y, dim, ii)
            
            for jj = 1:ysize2
                ytt = yt[jj]

                if ytt < ym[jj].min
                    ym[jj].min = ytt
                end

                if ytt > ym[jj].max
                    ym[jj].max = ytt
                end
            end
        end
        end
        
        
        
        return PolyFit{elt}(collect(v ^ p for v in x, p in 0:deg) \ y,
                            deg, scaled, xs, ys)
    else
        return PolyFit{elt}(collect(v ^ p for v in x, p in 0:deg) \ y,
                            deg, scaled, nothing, nothing)
    end
end

# module
end
