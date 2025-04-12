export gen_trial

# TODO: generalize
function gorilla_move(g:: SVector{2,Float64})
    g - [4, 0]
end

"""
sample a random trial with n timesteps and then output it within a dictionary format.
"""
function gen_trial(wm::SchollWM, n::Int64) # TODO: generalize: initial cond and movement
    @unpack area_height, area_width, dot_radius = wm
    _, states = wm_scholl(n, wm)
    delta = 2 * dot_radius
    bbmin = S2V(-area_width * 0.5 + delta,
                -area_height * 0.5 + delta)
    bbmax = S2V(area_width * 0.5 - delta,
                area_height * 0.5 - delta)
    gorilla_x = uniform(620.0, 650.0)
    gorilla = SVector(gorilla_x, 240.0)
    result = Vector{Dict}(undef,n)
    for j in 1:n # frames
        state = states[j]
        objects = state.objects
        no = length(objects)
        onscreen = inbounds(gorilla, bbmin, bbmax)
        ns = onscreen ? no + 1 : no
        positions = Vector{SVector{2, Float64}}(undef, ns) # dots
        gorilla = gorilla_move(gorilla)

        for k in 1:no # amount of objects and their locations
            obj = objects[k]  # Accessing the number of objects to iterate within the nested loop
            positions[k] = obj.pos  # this stores the object positions within k
        end
        if onscreen
            positions[no + 1] = gorilla  # Store gorilla's position at the end of positions array
        end
        result[j] = Dict(:positions => positions) # dictionary language is within keys and values
    end

    return result # returning a dictionary with all the needed information for dots and gorilla
end
