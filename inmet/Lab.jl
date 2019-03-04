__precompile__(true)

module Lab

using ConfParser

export workspace_init, activate, deactivate, save

mutable struct Workspace
    config::ConfParse
    path::String
end

workspace = nothing


function activate(path::AbstractString)
    global workspace
    
    workspace = Workspace(ConfParse(path, "ini"), path)
end


function save(arr::Array, path::AbstractString, alias::AbstractString = "")
    global workspace
    
    conf = workspace.config
    
    commit!(conf, path, Dict())
    
    commit!(conf, path, "type", "Array")
    commit!(conf, path, "dtype", "a")
    commit!(conf, path, "dim", "$(ndims(arr))")
    commit!(conf, path, "shape", "$(size(arr))")
    
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
