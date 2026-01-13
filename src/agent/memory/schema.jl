
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
    bell_idx_to_gstring(g.nrep, g.node_id)
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
    Int(object_count(t)) == g.nrep
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
        gs = shift_growth_indices(gs, birth_idx)
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

    (idx < 0 || n < (idx + 1)) && error("Shift index must be between 1 and n")
    
    new_gs = Vector{Int}(undef, n+1)

    max_so_far = -1
    for i = 1:n
        x = gs[i]
        if i < idx
            max_so_far = max(max_so_far, x)
            new_gs[i] = x
        else
            new_gs[i + 1] = x <= max_so_far ? x : x + 1
        end
    end
    new_gs[idx] = max_so_far + 1
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

    display(cm)
    print_granularity_schema(tr)
    # No change
    cm[:s0 => :nsm] == 1 &&  return g

    # Split k'th ensemble
    cm[:s0 => :nsm] == 2 &&
        return split_schema(g, cm[:s0 => :state => :idx])

    # Merge a and b
    ns = single_count(tr)
    nr = representation_count(tr)
    merge_schema(g, cm[:s0 => :state => :pair], ns, nr)
end

function split_schema(g::GranularitySchema, k::Int)
    gs = growth_string(g)
    gidx = find_k_block(gs, k)
    println("Before split; $(gs)")
    gs = split_growth_string(gs, gidx)
    println("After split; $(gs)")
    GranularitySchema(g.nrep, growth_to_index(gs, g.nrep))
end

function merge_schema(g::GranularitySchema,
                      pair_idx::Int,
                      ns::Int, nr::Int)
    gs = growth_string(g)
    # In rep space
    a, b = combination(nr, 2, pair_idx)
    @show (a,b)
    println("Before merge: $(gs)")
    a_idx = rep_to_partition_head(ns, gs, a)
    b_idx = rep_to_partition_head(ns, gs, b)
    anchor = min(a_idx, b_idx)
    mast = max(a_idx, b_idx)
    gs = merge_growth_string(gs, anchor, mast)
    println("After merge: $(gs)")
    GranularitySchema(g.nrep, growth_to_index(gs, g.nrep))
end

function rep_to_partition_head(ns::Int, gs::Vector{Int}, x::Int)
    if x <= ns
        # Find the x'th individual rep
        find_k_singleton(gs, x)
    else
        # Find the x'th ensemble rep
        find_k_block(gs, x - ns)
    end 
end

# Parition set 35: [0, 1, 1, 2, 3]
# Repr indices:    {1, 4, 5, {2,3}}
# + Merge(2, 4)
#
# 1. find `2` -> 
# Expected result: [0, 1, 1, 1, 2]
#   {1, 5, {2,3,4}}
#   
# TRG [0, 1, 2, 0, 1]
# IDX [1, 2, 3, 4, 5]
#
# TRG [0, 0, 1, 2, 1]
# IDX [1, 2, 3, 4, 5]
"""
    find_k_singleton(gs::Vector{Int}, k::Int)

Finds the index in the growth string of the k'th singleton block.
A singleton block is a block (label) that appears exactly once in the growth string.
The singletons are ordered by their position in the string.
For example, gs = [0, 1, 2, 1, 3] and k = 2 should return 3.
This implementation allocates O(max(gs)) space for frequency counts.
"""
function find_k_singleton(gs::Vector{Int}, k::Int)
    n = length(gs)
    if n == 0 || k < 1
        error("Invalid input")
    end
    m = maximum(gs)
    freq = zeros(Int, m + 1)
    for a in gs
        freq[a + 1] += 1
    end
    count = 0
    for i = 1:n
        if freq[gs[i] + 1] == 1
            count += 1
            count == k && return i
        end
    end
    error("Less than $(k) singletons in $(gs)")
end

"""
    find_k_block(gs::Vector{Int}, k::Int)

Finds the index in the growth string of the first element of the k'th multi-element block.
A multi-element block is a block (label) that appears more than once in the growth string.
The blocks are ordered by the position of their first appearance in the string.
For example, gs = [0, 1, 2, 1, 1] and k = 1 should return 2.
This implementation allocates O(max(gs)) space for frequency counts and seen flags.
"""
function find_k_block(gs::Vector{Int}, k::Int)
    n = length(gs)
    if n == 0 || k < 1
        error("Invalid input")
    end
    m = maximum(gs)
    freq = zeros(Int, m + 1)
    for a in gs
        freq[a + 1] += 1
    end
    count = 0
    seen = falses(m + 1)
    for i = 1:n
        lbl = gs[i] + 1
        if !seen[lbl] && freq[lbl] > 1
            seen[lbl] = true
            count += 1
            if count == k
                return i
            end
        end
    end
    error("Less thank $(k) multi-element blocks in: $(gs)")
end
"""
    merge_growth_string(gs::Vector{Int}, a::Int, b::Int)

Given the first indices of two blocks (of any length), produce a new growth string that merges them (they share the same integer label).
Modifies a copy of gs and returns it.
Assumes a and b are the minimal indices (first occurrences) of two different blocks.
Merges by assigning the smaller label to the larger label's positions, then decrementing all higher labels to fill the gap.
For example, gs=[0, 1, 1, 2, 3], a=2, b=4, yields [0, 1, 1, 1, 2].
This implementation allocates O(n) space for the new vector but avoids additional allocations inside.
"""
function merge_growth_string(gs::Vector{Int}, a::Int, b::Int)
    n = length(gs)
    if !(1 ≤ a ≤ n && 1 ≤ b ≤ n && a ≠ b)
        error("Invalid indices")
    end
    lab1 = gs[a]
    lab2 = gs[b]
    if lab1 == lab2
        error("Indices must point to different blocks")
    end
    keep = min(lab1, lab2)
    replace = max(lab1, lab2)
    new_gs = copy(gs)
    for i = 1:n
        if new_gs[i] == replace
            new_gs[i] = keep
        elseif new_gs[i] > replace
            new_gs[i] -= 1
        end
    end
    return new_gs
end

"""
    split_growth_string(gs::Vector{Int}, x::Int)

Given the first index of a block (of length >= 2), produce a new growth string that splits the block.
Assumes that the block is split such that the element at x remains in the original block (as a singleton if necessary),
and all subsequent elements in the same block are moved to a new block with label original+1, shifting higher labels up.
For example, gs=[0, 1, 1, 1, 2], x=2, yields [0, 1, 2, 2, 3].
This implementation allocates O(n) space for the new vector but avoids additional allocations inside.
Throws an error if the block at x has length < 2.
"""
function split_growth_string(gs::Vector{Int}, x::Int)
    n = length(gs)
    if !(1 ≤ x ≤ n)
        error("Invalid index")
    end
    u = l = gs[x]
    next = 0
    for i = x+1:n
        if gs[i] == l
            next = i
            break
        end
        u = max(u, gs[i])
    end
    next === 0 && error("Size of block at $(x) for $(gs) is 1")

    shift = next == (x+1) ? 1 : u - l
    new_gs = copy(gs)
    for i = next:n
        if new_gs[i] == l
            new_gs[i] = u + 1
        elseif u < new_gs[i]
            new_gs[i] += shift
        end
        # Don't do anything if z <= u
    end
    return new_gs
end

# Q: [0, 1, 2, 3, 4, 5, 4, 4]
# A: [0, 1, 2, 3, 4, 5, 6, 6]

# Q: [0, 1, 2, 3, 4, 5, 4, 6]
# A: [0, 1, 2, 3, 4, 5, 6, 7]

# Q: [0, 1, 2, 1, 1, 3, 4, 4]
# A: [0, 1, 2, 3, 3, 4, 5, 5]
