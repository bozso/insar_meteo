__precompile__(true)

module Lab

using ConfParser

export workspace_init, activate, deactivate, save

mutable struct Workspace
    config::ConfParse
    path::String
end

workspace = nothing

const str2type = Dict(
    "Float32"=>Float32,
    "Float64"=>Float64,


function activate(path::AbstractString)
    global workspace
    
    workspace = Workspace(ConfParse(path, "ini"), path)
end


function save(arr::Array, path::AbstractString, alias::AbstractString = "")
    global workspace
    
    conf = workspace.config
    
    commit!(conf, path,
            Dict("type"=>"Array",
                 "dtype"=>"$(eltype(arr))",
                 "ndims"=>"$(ndims(arr))",
                 "shape"=>"$(size(arr))"
                )
            )
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
