
"A set of moves that map ALPHA to NOW"
struct RepSpec
    "Steps applied in order from 1->k"
    steps::Vector{GranularityMove}
end

"""
    $(TYPEDSIGNATURES)

A granularity schema for the space of `N` individuals.

$(TYPEDFIELDS)
"""
struct GranularitySchema
    "Number of "
    nrep::Int
    "Graphical location of schema in `N`"
    node_id::Int
end

function growth_string(g::GranularitySchema)
    bell_idx_to_gstring(g.rep, g.node_id)
end

function alpha_schema(t::InertiaTrace, nrep::Int)
    GranularitySchema(nrep, bell_number(nrep))
end

function omega_schema(t::InertiaTrace, nrep::Int)
    GranularitySchema(nrep, 1)
end

function is_valid_schema(t::InertiaTrace, g::GranularitySchema)
    representation_count(t) == s.nrep
end

function ammend_schema(t::InertiaTrace, g::GranularitySchema)
    gs = growth_string(g)
    rc = representation_count(t)
    if g.nrep < rc
        # Death - one less single
        death_idx = find_death_idx(t) # TODO
        popat!(gs, death_idx)
    else
        # Birth - one more single
        birth_idx = single_count(t)
        gs = shift_growth_indices!(gs, birth_idx)
    end
    new_bell_idx = growth_to_index(gs, rc)
    GranularitySchema(rc, new_bell_idx)
end

function find_death_idx(trace::InertiaTrace)
    t, wm, istate = get_args(trace)
    death_idx = 0
    for k = 1:t
        # death occured here
        if trace[:kernel => k => :bd => :i] == 3
            death_idx = trace[:kernel => k => :bd => :switch => :dead]
        end
    end
    death_idx === 0 && error("No death found in $(get_choices(trace))")
    return death_idx
end

function shift_growth_indices(gs::Vector{Int}, idx::Int)
    n = length(gs)
    new_gs = Vector{Int}(undef, n)

    @inbounds for i = 1:n
        shift = ifelse(i < idx, 0, 1)
        new_gs[i + shift] = gs[i] + shift
    end
    new_gs[idx] = gs[idx]
    return new_gs
end
