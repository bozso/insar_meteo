__precompile__(true)

module PolynomFit

export PolyFit, Scale, fit_poly

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
    nfit::Int64
    scaled::Bool
    scales::Union{Vector{Scale{T}}, Nothing}

    """
    scale = x_max - x_min
    
    scaled(x)      = (x - x_min) / scale
    scaled^(-1)(x) =  scaled(x) * scale  + x_min
    
    x = 1.0
    f(x_min) = 0.0
    """
end


function Base.show(io::IO, p::PolyFit{T}) where T<:Number
    print(io, "PolyFit{$T} Fit degree: $(p.deg) Fitted Coefficients: ",
              "$(p.coeffs) Number of fits: $(p.nfit) ")
    
    if p.scaled
        print(io, "Scaled: true, Scales: $(p.scales)\n")
    else
        print(io, "Scaled: false")
    end
end


function fit_poly(x::Vector{T}, y::VecOrMat{T}, deg::Int64,
                  scaled::Bool=true, dim::Int64=1) where T<:Number
    
    # TODO: make it general for y::Array{T,N}; transpose when dim == 1
    dim <= 0 && error("dim should be > 0!")
    
    n = ndims(y)
    
    @inbounds begin
    
    if n == 1
        dim > 1 && error("")
        yy = y
    elseif n == 2
        if dim == 1
            yy = y
        elseif dim == 2
            yy = transpose(y)
        else
            error("dim >= 2")
        end
    else
        error("Max 2 dim.")
    end

    nfit = size(yy, 2)

    if scaled
        search = hcat(x, yy)
        
        minmax = extrema(search, dims=1)
        
        scales = vec([Scale(m) for m in minmax])
        
        # f(x) = (x - x_min) / (x_max - x_min)
        
        xs = scales[1]
        design = collect( ((xx - xs.min) / xs.scale)^p for xx in x, p in deg:-1:0)
        
        if n == 1
            ys = scales[2]
            @simd for jj in eachindex(yy)
                yy[jj] = (yy[jj] - ys.min) / ys.scale
            end
        else
            rows, cols = size(yy)
            for jj in 1:cols
                ys = scales[jj + 1]
                @simd for ii in 1:rows
                    iscale = 1.0 / ys.scale
                    yy[ii,jj] = (yy[ii,jj] - ys.min) * iscale
                end
            end
        end
    else
        yy, xs, ys = y, nothing, nothing
        design = collect(xx^p for xx in x, p in deg:-1:0)
    end
    
    # @inbounds
    end
    
    return PolyFit{T}(design \ yy, deg, nfit, scaled, scales)
end


function eval_poly(p::PolyFit{T}, x::T) where T<:Number
    nfit, deg, coeffs = p.nfit, p.deg, p.coeffs
    
    ret = Vector{T}(undef, nfit)
    
    @inbounds begin
    
    if p.scaled
        scales = p.scales
        xs = scales[1]
        
        xx = (x - xs.min) / xs.scale
        if deg == 1
            for ii in 1:nfit
                ys = scales[ii + 1]
                ret[ii] = (coeffs[1, ii] * xx + coeffs[2, ii]) * ys.scale + ys.min
            end
            return ret
        else
            for jj in 1:nfit
                ret[jj] = coeffs[1, jj] * xx
                
                for ii in 2:deg
                    ret[jj] += coeffs[ii, jj] * xx
                end

                ys = scales[jj + 1]
                ret[jj] = (ret[jj] + coeffs[deg + 1, jj]) * ys.scale + ys.min
            end
            return ret
        # if deg == 1
        end
    else
        if deg == 1
            for ii in 1:nfit
                ret[ii] = coeffs[1, ii] * x + coeffs[2, ii]
            end
            return ret
        else
            for jj in 1:nfit
                ret[jj] = coeffs[1, jj] * x
                
                for ii in 2:deg
                    ret[jj] += coeffs[ii, jj] * x
                end
            end
            return ret
        # if deg == 1
        end
    # if p.scaled
    end
    
    # @inbounds
    end
# eval_poly
end


function eval_poly(p::PolyFit{T}, x::Vector{T}) where T<:Number
    nfit, nx, deg, coeffs = p.nfit, length(x), p.deg, p.coeffs
    
    ret = Matrix{T}(undef, nfit, nx)
    
    @inbounds begin
    
    if p.scaled
        scales = p.scales
        xs = scales[1]
        
        if deg == 1
            for jj in 1:nx
                xx = (x[jj] - xs.min) / xs.scale
                for ii in 1:nfit
                    ys = scales[ii + 1]
                    ret[ii, jj] = (coeffs[1, ii] * xx + coeffs[2, ii]) * ys.scale + ys.min
                end
            end
            return ret
        else
            for jj in 1:nx
                xx = (x[jj] - xs.min) / xs.scale
                
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, jj] * xx
                end
                
                for ii in 1:nfit
                    for kk in 2:deg
                        ret[ii, jj] += coeffs[kk, ii] * xx
                    end
                end
                
                ys = scales[jj + 1]
                
                for ii in 1:nfit
                    ret[ii, jj] = (ret[ii, jj] + coeffs[deg + 1, jj]) * ys.scale + ys.min
                end
            end
            return ret
        # if deg == 1
        end
    else
        if deg == 1
            for jj in 1:nx
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, ii] * x + coeffs[2, ii]
                end
            end
            return ret
        else
            for jj in 1:nx
                for ii in 1:nfit
                    ret[ii, jj] = coeffs[1, jj] * x
                end
                
                for ii in 1:nfit
                    for kk in 2:deg
                        ret[ii, jj] += coeffs[kk, ii] * x
                    end
                end
            end
            return ret
        # if deg == 1
        end
    # if p.scaled
    end
    
    # @inbounds
    end
    
# eval_poly
end


function (p::PolyFit{T})(x::Union{T,Vector{T}}) where T<:Number
    return eval_poly(p, x)
end


function scale_back(p::Scale{T}, x::T) where T<:Number
    return x * p.scale + p.min
end


function scale_back(p::Scale{T}, x::Vector{T}) where T<:Number
    n, min, scale = length(x), p.min, p.scale
    xx::Vector{T}(undef, n)
    
    @inbounds @simd for ii = 1:n
        xx[ii] = x[ii] * scale + min
    end
    
    return xx
end

function scale_it(p::Scale{T}, x::T) where T<:Number
    return (x - p.min) / p.scale
end


function scale_it(p::Scale{T}, x::Vector{T}) where T<:Number
    n, min, iscale = length(x), p.min, 1.0 / p.scale
    xx::Vector{T}(undef, n)
    
    @inbounds @simd for ii = 1:n
        xx[ii] = (x[ii] - min) / iscale
    end
    
    return xx
end


# module
end
