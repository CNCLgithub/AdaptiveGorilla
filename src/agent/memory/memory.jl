

abstract type GranularityProtocal end

abstract type GranularityMove end

struct SplitMove <: GranularityMove
    x::Int64
end

struct MergeMove <: GranularityMove
    a::Int64
    b::Int64
end

const SplitMergeMove = Union{SplitMove, MergeMove}

"""
Defines the granularity mapping for a current trace format
"""
@with_kw struct AdaptiveGranularity <: GranularityProtocal
    """Ratio between objectve 1 and 2"""
    m::Float64
    """Number of particles to approx split-merge alpha"""
    alpha_steps::Int64
    """Number of objects to evaluate"""
    object_steps::Int64
end

function granularity_objective(ag::MentalModule{G},
                               ac::MentalModule{A}
                               chain::APChain,
    ) where {G<:AdaptiveGranularity, A<:AdaptiveComputation}
    attp, attx = parse(ac)
    @unpack partition, nns = attp
    @unpack state = chain
    # Running averages
    mag = -Inf
    len = 0.0
    np = length(state)
    @inbounds for i = 1:np
        tr = task_relevance(attx,
                            attp.partition,
                            state.traces[i],
                            attp.nns)
        mag = logsumexp(mag, l2log(tr))
        len += length(tr)
    end
    mag - log(len)
end

function shift_granularity(t::T, m::SplitMergeMove) where {T<:InertiaTrace)}
    _, wm, _ = get_args(t)
    state = get_last_state(t)
    @unpack singles, ensembles = state
    cm = choicemap()
    if rand() > 0.5
        ns = length(singles)
        ne = length(ensemble)
        sample_granularity_move!(cm, ns, ne)
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

function sample_granularity_move!(cm<:Gen.ChoiceMap, nsingle::Int, nensemble::Int)
    ntotal = nsingle + nensemble
    nmerges = ncr(ntotal, 2)
    move = if nensemble > 0 && rand() > 0.5
        # sample which ensemble to split
        cm[:s0 => :nsm] = 2
        cm[:s0 => :state => :idx] = rand(1:nensemble)
    else
        # sample which pair to merge
        cm[:s0 => :nsm] = 3
        cm[:s0 => :state => :pair] = rand(1:nmerges)
    end
    return nothing
end

function apply_granularity_move(wm::InertiaWM, state::InertiaState, m::MergeMove)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    a = object_from_idx(state, m.a)
    b = object_from_idx(state, m.b)
    e = apply_merge(a, b)
    # need to determine what object types were merged
    delta_ns = 0
    delta_ns += isa(a, InertiaSingle) ? -1 : 0
    delta_ns += isa(b, InertiaSingle) ? -1 : 0
    delta_ne = 1
    delta_ne += isa(a, InertiaEnsemble) ? -1 : 0
    delta_ne += isa(b, InertiaEnsemble) ? -1 : 0
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)
    # Remove any merged singles
    c = 1
    for i = 1:ns
        (isa(a, InertiaSingle) && i == m.a) && continue
        (isa(b, InertiaSingle) && i == m.b) && continue
        new_singles[c] = singles[i]
    end
    # Remove merged ensembles
    c = 1
    for i = 1:ne
        (isa(a, InertiaEnsemble) && i == m.a) && continue
        # replace `b` with `e`
        # (keep new ensembles towards the end of the array)
        if (isa(b, InertiaEnsemble) && i == m.b)
            new_ensembles[c] = e
        else
            new_singles[c] = singles[i]
        end
        c += 1
    end
    InertiaState(new_singles, new_ensembles)
end

function apply_granularity_move(wm::InertiaWM, state::InertiaState, m::SplitMove)
    @unpack singles, ensembles = state
    ns = length(singles)
    ne = length(ensembles)
    e = ensembles[m.x]
    # if the ensemble is a pair, we get two individuals
    delta_ns = e.rate == 2 ? 2 : 1
    delta_ne = e.rate == 2 ? -1 : 0
    new_singles = Vector{InertiaSingle}(undef, ns + delta_ns)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne + delta_ne)
    
    new_singles[1:ns] = singles
    if e.rate == 2 # E -> (S, S)
        (a, b) = sample_split_pair(wm, x)
        new_singles[ns + 1] = a
        new_singles[ns + 2] = b
        c = 1
        for i = 1:ne
            i == m.x && continue
            new_ensembles[c] = ensembles[i]
            c += 1
        end
    else # E -> (E', S)
        s = sample_split(wm, x)
        e = apply_split(e, s)
        new_singles[ns + 1] = s
        new_ensembles[:] = ensembles[1:ne]
        new_ensembles[m.x] = e
    end

    InertiaState(new_singles, new_ensembles)
end


function regranularize!(mem::MentalModule{M},
                        att::MentalModule{A},
                        vis::MentalModule{V}
    ) where {M<:AdaptiveGranularity,
             A<:AdaptiveComputation,
             V<:HyperFilter}

    # Weight granularity objective
    hf, vstate = parse(vis)
    ws = zeros(hf.h)
    for i = 1:hf.h
        ws[i] = granularity_objective(ag, vstate.chains[i], astate)
    end
    # Repopulate and shift granularity
    ws = softmax(ws)
    next_gen = multinomial(ws)
    new_chains = Vector{APChain}(undef, hf.h)
    for i = 1:hf.h
        parent = vstate.chains[next_gen[i]] # PFChain
        template = retrieve_map(parent) # InertiaTrace
        cm = shift_granularity(template) # InertiaState
        new_chains[i] = reinit_chain(parent, cm)
    end

    vstate.age = 0 # TODO: 0 or 1? 
    vstate.chains = new_chains
    
end

function foo(chain::APChain, p::AdaptiveComputation, g::AdaptiveGranularity)
    # Get object weights
    trs = compute_aligned_task_relevance(chain, p) # TODO

    # greedy approach, look at lowest tr first
    local m::SplitMergeMove, w::Float64
    toconsider = sortperm(trs)[1:g.object_steps]
    # merges
    for i = toconsider[1:(g.object_steps-1)]
        for j = toconsider[2:end]
            _m = Merge(i, j)
            _w = gratio(m, chain, i, trs[i], j, trs[j])
            if w < _w
                m = _m
                w = _w
            end
        end
    end
    # splits
    for i = toconsider
        if cansplit(current_schema, i)
            _m = Split(i)
            _w = gratio(s, chain, i, trs[i])
            if w < _w
                m = _m
                w = _w
            end
        end
    end



    # split-merge
    merge_weights = compute_merge_weights()
    to_merge = resolve_merge(merge_weights, nabla)

    split_weights = compute_split_weights(to_merge)
    to_split = resolve_split(split_weights, nabla)




end
