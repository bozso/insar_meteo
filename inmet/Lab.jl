__precompile__(true)

module Lab

using ConfParser

export workspace_init, activate, deactivate, save, load

mutable struct Workspace
    config::ConfParse
    path::String
end

workspace = nothing

const Astr = AbstractString

const valid_types = (Float32, Float64, Int64)

const str2dtype = Dict((("$(dtype)", dtype)
                       for dtype in valid_types))


function activate(path::AbstractString)
    global workspace
    
    workspace = Workspace(ConfParse(path, "ini"), path)
end


function save(arr::Array, name::Astr, path::Astr)
    global workspace
    
    etype = eltype(arr)

    commit!(workspace.config, name,
            Dict("type"=>"Array",
                 "dtype"=>"$(etype)",
                 "ndims"=>"$(ndims(arr))",
                 "shape"=> join(("$elem" for elem in size(arr)), ";"),
                 "path"=>path
                )
            )
    
    open(path, "w") do f
        for elem in arr
            write(f, elem)
        end
    end
end


function load(name::Astr)
    global workspace, str2dtype

    tp = retrieve(workspace.config, name, "type")
    dt = str2dtype[retrieve(workspace.config, name, "dtype")]
    nd = retrieve(workspace.config, name, "ndims", Int64)
    shape = string(retrieve(workspace.config, name, "shape"))
    path = retrieve(workspace.config, name, "path")
    
    shape = [parse(Int64, elem) for elem in split(shape, ";")]
    
    # _arr = Vector{dt}(undef, prod(shape))
    
    open(path, "r") do f
        _arr = reinterpret(dt, read(f))
    end
    
    return reshape(_arr, shape)
end


function deactivate(path::Union{AbstractString,Nothing} = nothing)
    global workspace
    
    workspace == nothing && error("No workspace is active!")
    
    if path != nothing
        workspace.path = path
    end
    
    save!(workspace.config, workspace.path)
end

# module
end
