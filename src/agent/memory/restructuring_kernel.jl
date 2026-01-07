export StaticRKernel, SplitMergeKernel,
    UniformSplitMerge, MhoSplitMerge

using SparseArrays: sparsevec

################################################################################
# Dummy Static Kernel
################################################################################

"No restructuring"
struct StaticRKernel <: RestructuringKernel end

function restructure_kernel(::StaticRKernel, t::InertiaTrace)
    cm = choicemap()
    cm[:s0 => :nsm] = 1 # no change
    return cm
end

################################################################################
# Split-merge Kernel
################################################################################

abstract type SplitMergeKernel <: RestructuringKernel end

function restructure_kernel end


################################################################################
# Split-merge Heuristics
################################################################################


@with_kw struct UniformSplitMerge <: SplitMergeKernel
    restructure_prob::Float64 = 0.5
end

function restructure_kernel(kappa::UniformSplitMerge,
                            t::InertiaTrace)
    cm = choicemap()
    if rand() < kappa.restructure_prob
        # SPLIT | MERGE
        if rand() < split_prob(kappa, t)
            sample_split_move!(cm, kappa, t)
        else
            sample_merge_move!(cm, kappa, t)
        end
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

function split_prob(::UniformSplitMerge, tr::InertiaTrace)
    # Only one ensemble -> can't merge
    representation_count(tr) == 1 && return 1.0
    # No ensemble -> can't split
    ensemble_count(tr) == 0 && return 0.0
    # 50/50 split/merge
    return 0.5
end

function sample_split_move!(cm::ChoiceMap,
                            h::UniformSplitMerge,
                            t::InertiaTrace)
    nensemble = ensemble_count(t)
    cm[:s0 => :nsm] = 2 # split branch
    cm[:s0 => :state => :idx] = rand(1:nensemble)
    return nothing
end

function sample_merge_move!(cm::ChoiceMap,
                            h::UniformSplitMerge,
                            t::InertiaTrace)

    nsingle = single_count(t)
    nensemble = ensemble_count(t)
    ntotal = nsingle + nensemble
    nmerges = ncr(ntotal, 2)
    cm[:s0 => :nsm] = 3 # merge branch
    cm[:s0 => :state => :pair] = rand(1:nmerges)
    return nothing
end

@with_kw struct MhoSplitMerge <: SplitMergeKernel
    "Reference to and AdaptiveComputation module"
    att::MentalModule{<:AdaptiveComputation}
    # Restructure
    restructure_prob_min::Float64 = 0.25
    restructure_prob_max::Float64 = 0.75
    restructure_prob_delta::Float64 = (restructure_prob_max -
        restructure_prob_min)
    restructure_prob_slope::Float64 = 10.0
    # Split
    split_tau::Float64 = 1.0
    # Merge
    merge_tau::Float64 = 20.0
    "Maximum element count considered for merging"
    merge_max_elems::Int64 = 5
end

function restructure_kernel(kappa::MhoSplitMerge,
                            t::InertiaTrace)
    # Task-relevance will inform the kernel at several points
    attp, attx = mparse(kappa.att)
    tr = task_relevance(attx,
                        attp.partition,
                        t,
                        attp.nns)

    cm = choicemap()
    if rand() < restructure_prob(kappa, tr)
        smw = split_prob(kappa, t, tr)
        # SPLIT | MERGE
        # if rand() < split_prob(kappa, t, tr)
        # println("Pr(Split) = $(smw)")
        if rand() < smw
            sample_split_move!(cm, kappa, t, tr)
        else
            sample_merge_move!(cm, kappa, t, tr)
        end
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
end

function restructure_prob(k::MhoSplitMerge, tr::Vector{Float64})
    mag = logsumexp(tr) # REVIEW: needed elsewhere? 
    x = exp(mag / k.restructure_prob_slope)
    w = k.restructure_prob_min + min(k.restructure_prob_delta, x)
    # println("Restructure prob: $(w)")
    return w
end

function split_prob(kappa::MhoSplitMerge,
                    tr::InertiaTrace,
                    deltas::Vector{Float64})
    ns = single_count(tr)
    ne = ensemble_count(tr)
    re = representation_count(tr)
    # Edge cases:
    #   (1) No ensemble -> can't split
    #   (2) Only one ensemble -> can't merge
    ne == 0 && return 0.0
    re == 1 && return 1.0

    # Guess whether split or merge is better.
    # Possible heuristics:
    #   1. more than 1 -Inf? => Merge
    #   2. Max delta is ensemble? => Split
    count(isinf, deltas) > 2  && return 0.1
    argmax(deltas) > ns && return 0.9
     
    # 50/50
    return 0.5
end

function sample_split_move!(cm::ChoiceMap,
                            h::MhoSplitMerge,
                            t::InertiaTrace,
                            deltas::Vector{Float64})
    ws = split_weights(h, t, deltas)
    # println()
    # print_granularity_schema(t)
    # @show ws
    cm[:s0 => :nsm] = 2 # split branch
    # indices are already in ensemble space
    cm[:s0 => :state => :idx] = categorical(ws)
end

function sample_merge_move!(cm::ChoiceMap,
                            h::MhoSplitMerge,
                            t::InertiaTrace,
                            deltas::Vector{Float64})
    ws = merge_weights(h, t, deltas)
    # println()
    # print_granularity_schema(t)
    # n = representation_count(t)
    # display(Dict(zip(map(x -> combination(n, 2, x), ws.nzind), ws.nzval)))
    # error()
    pair_idx = categorical(ws)
    # pair_idx = argmax(ws)
    cm[:s0 => :nsm] = 3 # merge branch
    cm[:s0 => :state => :pair] = pair_idx
    return nothing
end

function split_weights(k::MhoSplitMerge,
                       t::InertiaTrace,
                       deltas::Vector{Float64})
    ne = ensemble_count(t)
    ne == 1 ? [1.0] : softmax(deltas[end-ne+1:end], k.split_tau)
end

function merge_weights(k::MhoSplitMerge,
                       t::InertiaTrace,
                       deltas::Vector{Float64})

    # Determine importance: Less importance -> higher merge weight
    # NOTE: importance temperature scales with |tr|
    #   - at low |tr|, differences don't matter as much
    # temp = 10*attp.itemp - logsumexp(tr)
    # temp = max(temp, 1.0)
    # importance = softmax(tr, temp)
    # importance = softmax(tr, attp.itemp)

    attp, attx = mparse(k.att)
    # @show tr
    # @show importance
    # The weight of each merge pair is simply the sum of their importance values
    ntotal = length(deltas)
    # Only consider the `k` least important elements
    # (to reduce combinatoric explosions)
    ncandidates = min(ntotal, k.merge_max_elems)
    cand_indices = sort(partialsortperm(deltas, 1:ncandidates))
    sparse_pairs = ncr(ncandidates, 2)
    pair_id = Vector{Int64}(undef, sparse_pairs)
    pair_ws = Vector{Float64}(undef, sparse_pairs)
    for i = 1:sparse_pairs
        (x, y) = combination(ncandidates, 2, i)
        a = cand_indices[x]
        b = cand_indices[y]
        # Pr(Merge) inv. prop. importance
        pair_id[i] = combination_rank(ntotal, 2, [a, b])
        # pair_ws[i] = .75*(1.0 - importance[a])^75 * .75*(1.0 - importance[b])^75
        # pair_ws[i] = 0.5((1.0 - importance[a])^100 * (1.0 - importance[b])^100)
        pair_ws[i] = logsumexp(deltas[a], deltas[b]) +
            dissimilarity(t, attp.map_metric, a, b)
        # pair_ws[i] = logsumexp(
        #     logsumexp(deltas[a], deltas[b]),
        #     -dissimilarity(t, a, b))
        # println("W: $(a),$(b) => $(pair_ws[i])")
        # println("ID: $(a),$(b) => $(pair_id[i]) =>"*
        #     " $(combination(ntotal, 2, pair_id[i]))")
    end
    # deterministic if only 1 pair | categorical
    if ncandidates > 2
        # Normalize
        # rmul!(pair_ws, 1.0 / sum(pair_ws))
        pair_ws = inv_softmax(pair_ws, k.merge_tau)
    else
        pair_ws[1] = 1.0
    end
    
    # Store merge-weights in a sparse vector
    total_pairs = ncr(ntotal, 2)
    ws = sparsevec(pair_id, pair_ws, total_pairs)

    # NOTE: to retrieve members use:
    # selected = combination(total_pairs, 2, pair_idx)
    return ws
end
