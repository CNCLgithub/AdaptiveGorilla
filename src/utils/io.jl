
export load_json

"""
    load_json(path::String)::Dict{Symbol, Any}

Opens the file at path, parses as JSON and returns a dictionary
"""
function load_json(path::String)
    result = Dict{Symbol, Any}()
    open(path, "r") do f
        data = JSON3.read(f)
        for (k, v) in data
            result[Symbol(k)] = v
        end
    end
    return result
end

using TOML

"Links a mental module to another"
struct ModuleRef end

function link(; modules::Dict, ref::String)
    modules[Symbol(ref)]
end

function load_and_link!(subpart, parts, dest, proto, params)
    protocol = getfield(AdaptiveGorilla, Symbol(proto))
    subpart[Symbol(dest)] =
        if protocol <: ModuleRef
            link(; modules = parts, params...)
        else
            protocol(; params...)
        end
    return nothing
end

function load_inner(parts, toml)
    kwargs = Dict{Symbol, Any}()
    for (key, val) = toml
        if typeof(val) <: Dict && haskey(val, "protocol")
            proto = val["protocol"]
            # Recurse
            rest = haskey(val, "params") ? load_inner(parts, val["params"]) : Dict()

            # Nested component
            load_and_link!(kwargs, parts, key, proto, rest)
        else
            kwargs[Symbol(key)] = val
        end
    end
    return kwargs
end
