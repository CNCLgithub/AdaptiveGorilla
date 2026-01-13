
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

function alpha_schema(nrep::Int)
    GranularitySchema(nrep, bell_number(nrep))
end

function omega_schema(nrep::Int)
    GranularitySchema(nrep, 1)
end

# TODO: implement following
# - retrieve growth string
# - check whether it matches nsignles and nensembles
function is_valid_schema(t::InertiaTrace, g::GranularitySchema)
    representation_count(t) == g.nrep
end

function ammend_schema(t::InertiaTrace, g::GranularitySchema)
    gs = growth_string(g)
    rc = representation_count(t)
    if g.nrep < rc
        # Death - one less single
        death_idx = find_death_idx(t)
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

function guess_schema(tr::InertiaTrace)
    sc = single_count(tr)
    sc != representation_count(tr) &&
        error("Could not guess schema from InertiaTrace")

    alpha_schema(sc)
end


function memory_schema(perception::MentalModule{<:HyperFilter})
    visp, visstate = mparse(perception)
    chain = visstate.chains[1]
    trace = retrieve_map(chain)
    guess_schema(trace)
end

function transform_schema(tr::InertiaTrace, g::GranularitySchema, cm::ChoiceMap)
    t, _, _ = get_args(tr)

    # No change
    cm[:kernel => t => :s0 => :nsm] == 0 &&  return g

    # Split k'th ensemble
    cm[:kernel => t => :s0 => :nsm] == 1 &&
        split_schema(g, cm[:kernel => t => :s0 => :state => :idx])

    # Merge a and b
   merge_schema(g, cm[:kernel => t => :s0 => :state => :pair])
end

function split_schema(g::GranularitySchema, k::Int)
    gs = growth_string(g)
    find_k_dup(gs, gs[k], k)

end

# Parition set 35: [0, 1, 1, 2, 3]
# Repr indices:    {1, 4, 5, {2,3}}
# + Merge(2, 4)
# = [0, 1, 1, 1, 2]
#   {1, 5, {2,3,4}}
#   
function merge_schema(tr::InertiaTrace,
                      g::GranularitySchema,
                      pair_idx::Int)
    gs = growth_string(g)
    # In rep space
    a, b = combination(representation_count(tr), 2, pair_idx)
    a_idx = rep_to_part_index(tr, a)
    b_idx = rep_to_part_index(tr, b)
    pidx = findfirst()
    gs[b] = gs[a]
    GranularitySchema(g.nrep, growth_to_index(gs))
end

function rep_to_part_index(tr::InertiaTrace, x::Int)
    if x <= single_count(tr)
        
end

function scan_unique_k(a::Array, k::Int)
    na = length(a)
    cur_idx = 1
    cur_elem = first(a)
    nunique = 1
    @inbounds while cur_idx <= na
        

    end
end

function find_k_dup(gs::Vector{Int}, k::Int)
    i = 0
    c = 0
    n = length(gs)
    while i < n || c < k
        x = g[i]
        y = scan_forward(g, i+1, x)
        if (y - i) > 1
            c += 1
        end
        i += max(y, 1)
    end
    return i
end

function scan_forward(a::Array{T}, x::T,
                      start::Int=1) where {T}
    x = 0
    for i = s:length(a)
        if a[i] == x
            x = i
        else
            break
        end
    end
    return x
end
