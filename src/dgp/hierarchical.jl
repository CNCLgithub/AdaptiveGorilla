
struct HWM <: WorldModel
end

struct HWS <: WorldState{HWM}
    groups::Vector{Group}
    objects::Vector{Dot}
end

struct Group
    pos::S2V
    vel::S2V
    members::Vector{Int64}
end

members(g::Group) = g.members

function step(wm::HWM, ws::HWS,
              group_deltas::Vector{S2V},
              object_deltas::Vector{S2V})
    # Setup
    @unpack groups, objects = ws
    ng = length(groups)
    no = length(objects)
    new_groups = Vector{Group}(undef, ng)
    new_objects = Vector{Dot}(undef, no)
    # Update groups
    @inbounds for i = 1:ng
        group = groups[i]
        delta = group_deltas[i]
        new_groups[i] = update_elem(group, wm, delta)
        for m = members(group)
            object_deltas[m] = object_deltas[m] + delta
        end
    end
    # Update dots
    @inbounds for i = 1:no
        obj = objects[i]
        delta = object_deltas[i]
        new_objects[i] = update_elem(obj, wm, delta)
    end
    # Return updates elements
    HWS(new_groups, new_objects)
end
