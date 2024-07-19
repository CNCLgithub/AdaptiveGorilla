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
