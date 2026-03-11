using Statistics
using StaticArrays
using Statistics: mean
using LinearAlgebra: norm
using Accessors: setproperties
using MOTCore: scholl_init

################################################################################
# Metrics
################################################################################

struct Metrics
    funcs::Vector{Function}
    names::Vector{Symbol}

    function Metrics(fs, ns)
        @assert length(fs) == length(ns)
        new(fs, ns)
    end
end

function (ms::Metrics)(args::Tuple)
    n = length(ms.funcs)
    vs = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        vs[i] = ms.funcs[i](args...)
    end
    return vs
end

mutable struct RunningStats
    mus::Vector{Float64}
    vrs::Vector{Float64}
    c::Int64
end

function RunningStats(ms::Metrics)
    n = length(ms.names)
    RunningStats(zeros(n), zeros(n), 1)
end

function stride!(stats::RunningStats, ms::Metrics, args::Tuple)
    for mi = eachindex(ms.funcs)
        f = ms.funcs[mi]
        x = mean(f, args...)
        m_prev = stats.mus[mi]
        m = m_prev + ((x - m_prev) / stats.c)
        stats.vrs[mi] += (x - m_prev) * (x - m)
        stats.mus[mi] = m
    end
    stats.c += 1
end

function report(stats::RunningStats, ms::Metrics)
    n = length(ms.funcs)
    mus = Vector{Float64}(undef, n)
    sds = Vector{Float64}(undef, n)
    for mi = 1:n
        mus[mi] = stats.mus[mi]
        sds[mi] = sqrt(stats.vrs[mi] / (stats.c - 1))
    end

    Dict{Symbol, Tuple}(zip(ms.names, zip(mus, sds)))
end

function eccentricity(state::SchollState)
    # first 4 are targets
    objects = state.objects
    no = length(objects)
    val = 0.0
    @inbounds for i = 1:no
        # tdd = Inf
        val += norm(get_pos(objects[i]))
    end
    2 * (val / no)
end

function nearest_obj(state::SchollState, dmax::Float64 = 45.0)
    objects = state.objects
    d = Inf
    @inbounds for i = 1:7
        tpos = get_pos(objects[i])
        for j = (i+1):8
            d = min(norm(tpos - get_pos(objects[j])), d)
        end
    end
    min(d, dmax)
end

function tddensity(state::SchollState)
    # first 4 are targets
    objects = state.objects
    avg_tdd = 0.0
    @inbounds for i = 1:4
        tdd = Inf
        tpos = get_pos(objects[i])
        for j = 5:8
            d = norm(tpos - get_pos(objects[j]))
            avg_tdd += d
        end
    end
    avg_tdd / 16
end

function tdmin(state::SchollState)
    # first 4 are targets
    objects = state.objects
    tdd = Inf
    @inbounds for i = 1:4
        tpos = get_pos(objects[i])
        for j = 5:8
            # REVIEW: consider l1 distance
            d = norm(tpos - get_pos(objects[j]))
            tdd = min(tdd, d)
        end
    end
    tdd
end

################################################################################
# Math
################################################################################

function smooth(xs::Vector{Float64}, w::Int64)
    n = length(xs)
    ys = zeros(length(xs))
    w = isodd(w) ? w : w + 1
    hw = Int64((w + 1) / 2)
    @inbounds for i = 1:n
        c = 0
        for j = max(1, i - hw + 1):min(n, i + hw - 1)
            ys[i] += xs[j]
            c += 1
        end
        ys[i] *= 1.0 / c
    end
    return ys
end

################################################################################
# IO
################################################################################


function trial_to_dict(states::AbstractVector{T}) where {T<:SchollState}
    positions = map(states) do state
        map(state.objects) do dot
            get_pos(dot)
        end
    end
    Dict(:positions => positions)
end

function write_dataset(dataset::Vector, json_path::String)
    out = map(trial_to_dict, dataset)
    open(json_path, "w") do f
        write(f, json(out))
    end
    return nothing
end

function write_condlist(condlist::Vector, path::String)
    open(path, "w") do f
        write(f, json(condlist))
    end
    return nothing
end


################################################################################
# Gen
################################################################################

function burnin(wm::SchollWM, steps::Int64)
    state = scholl_init(wm)
    for t = 1:steps
        state = scholl_kernel(t, state, wm)
    end
    return state
end

function warmup(wm::SchollWM, metrics::Metrics,
                k::Int64=240, n::Int64=1000)
    stats = RunningStats(metrics)
    for _ = 1:n
        state = burnin(wm, 12)
        part = scholl_chain(k, state, wm)
        stride!(stats, metrics, (part,))
    end
    report(stats, metrics)
end
