export StaticRKernel, SplitMergeKernel,
    UniformSplitMerge, MhoSplitMerge

using SparseArrays: sparsevec

################################################################################
# Dummy Static Kernel
################################################################################

"No restructuring"
struct StaticRKernel <: RestructuringKernel end

function restructure_kernel(::StaticRKernel, ::Any, ::Any, t::InertiaTrace, i::Int)
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

function restructure_kernel(kappa::UniformSplitMerge, ::MemoryFitness, ::Nothing,
                            t::InertiaTrace, i::Int)
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
    ne = ensemble_count(tr)
    re = representation_count(tr)
    # Edge cases:
    #   (1) No ensemble -> can't split
    #   (2) Only one ensemble -> can't merge
    ne == 0 && return 0.0
    re == 1 && return 1.0
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
    restructure_prob::Float64 = 0.5
    # Split
    split_tau::Float64 = 1.0
    # Merge
    merge_tau::Float64 = 20.0
    "Maximum element count considered for merging"
    merge_max_elems::Int64 = 5
end

function restructure_kernel(kappa::MhoSplitMerge,
                            fitness::MhoFitness,
                            state::MhoScores,
                            t::InertiaTrace,
                            chain_idx::Int)
    # Task-relevance will inform the kernel at several points
    schema_id = state.schema_map[chain_idx]
    time_integral = reconstitute_deltas(state.rep_deltas,
                                        state.schema_registry,
                                        schema_id)
    cm = choicemap()
    if rand() < kappa.restructure_prob
        split_or_merge = split_prob(kappa, t, time_integral)
        if rand() < split_or_merge
            sample_split_move!(cm, kappa, t, time_integral)
        else
            sample_merge_move!(cm, kappa, t, time_integral)
        end
    else
        cm[:s0 => :nsm] = 1 # no change
    end
    return cm
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

    # If relatively few representations have all of the TR,
    # then recommend merging
    log_normed_deltas = deltas .- logsumexp(deltas)
    ess = Gen.effective_sample_size(log_normed_deltas)
    ess < 0.5 * re && return 0.1
     
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
    ntotal = length(deltas)
    if ntotal < 2
        error("Attempting to merge in trace with only 1 representation")
    end
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
        pair_id[i] = combination_rank(ntotal, 2, [a, b])
        pair_ws[i] = logsumexp(deltas[a], deltas[b])
    end
    if ncandidates > 2 
        pair_ws = inv_softmax(pair_ws, k.merge_tau, -1E5) # Normalize
    else 
        pair_ws[1] = 1.0 # deterministic if only 1 pair | categorical
    end
    # Store merge-weights in a sparse vector
    # NOTE: to retrieve members use:
    # selected = combination(total_pairs, 2, pair_idx)
    total_pairs = ncr(ntotal, 2)
    ws = sparsevec(pair_id, pair_ws, total_pairs)
    return ws
end
