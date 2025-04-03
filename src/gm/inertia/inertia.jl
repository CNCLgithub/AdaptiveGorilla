export InertiaWM, InertiaState

import MOTCore: get_pos

################################################################################
# Model definition
################################################################################

"""
Model that uses inertial change points to "explain" interactions
"""
@with_kw struct InertiaWM <: WorldModel

    # EPISTEMICS
    "Possible materials for objects. Sampled uniformly"
    materials::Vector{Material} = collect(instances(Material))
    "Average number of objects in the scene"
    object_rate::Float64 = 8.0
    "Number of individual representations"
    irate::Float64 = 0.5

    # Individual obj parameters
    single_size::Float64 = 10.0
    # Ensemble parameters
    ensemble_shape::Float64 = 2.0
    ensemble_scale::Float64 = 0.5

    "Probability that a baby is born for a given step"
    birth_weight::Float64 = 0.01


    # DYNAMICS
    # - general
    vel::Float64 = 10 # base vel
    area_width::Float64 = 400.0
    area_height::Float64 = 400.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)
    display_border::Float64 = 20.0
    # display region
    bbmin::S2V = S2V(-0.5 * area_width + display_border,
                     -0.5 * area_height + display_border)
    bbmax::S2V = S2V(0.5 * area_width - display_border,
                     0.5 * area_height - display_border)
    walls::SVector{4, Wall} = init_walls(area_width,
                                           area_height)

    # - inertia_force
    "Probability that an individual is stable for a given step"
    stability::Float64 = 0.9
    force_low::Float64 = 1.0
    force_high::Float64 = force_low * 5.0
    ensemble_var_shift::Float64 = 0.2

    # - wall force
    wall_rep_m::Float64 = 0.0
    wall_rep_a::Float64 = 0.02
    wall_rep_x0::Float64 = 0.0

    # Graphics
    single_noise::Float64 = 1.0
    material_noise::Float64 = 0.1
end

materials(wm::InertiaWM) = wm.materials

abstract type InertiaObject <: Object end

# Iso or bernoulli
@with_kw struct InertiaSingle <: InertiaObject
    mat::Material
    pos::S2V
    vel::S2V
    size::Float64 = 10.0
end

function InertiaSingle(m::Material, p::S2V, v::S2V)
    InertiaSingle(;mat = m,pos = p, vel = v)
end

get_pos(s::InertiaSingle) = s.pos
get_vel(s::InertiaSingle) = s.vel

# PPP
struct InertiaEnsemble <: InertiaObject
    rate::Float64
    matws::Vector{Float64}
    pos::S2V
    "Position variance"
    var::Float64
    vel::S2V
end

get_pos(e::InertiaEnsemble) = e.pos
get_vel(e::InertiaEnsemble) = e.vel
rate(e::InertiaEnsemble) = e.rate

struct InertiaState <: WorldState{InertiaWM}
    singles::AbstractVector{InertiaSingle}
    ensembles::AbstractVector{InertiaEnsemble}
end

function InertiaState(wm::InertiaWM,
                      singles::AbstractVector{InertiaSingle})
    # println("InertiaState ns=$(length(singles))")
    InertiaState(singles, InertiaEnsemble[])
end

function InertiaState(wm::InertiaWM,
                      singles::AbstractVector{InertiaSingle},
                      ensembles::AbstractVector{InertiaEnsemble})
    InertiaState(singles, ensembles)
end


function object_bounds(wm::InertiaWM)
    @unpack area_width, area_height = wm
    xs = S2V(-0.5*area_width, 0.5*area_width)
    ys = S2V(-0.5*area_height, 0.5*area_height)
    (xs, ys)
end

function force_prior(o::InertiaSingle, wm::InertiaWM)
    @unpack stability, force_low, force_high = wm
    (stability, force_low, force_high)
end

function force_prior(e::InertiaEnsemble, wm::InertiaWM)
    @unpack stability, force_low, force_high = wm
    @unpack rate = e
    (stability,
     force_low / rate,
     force_high / rate)
end


function step(wm::InertiaWM,
              state::InertiaState,
              supdates::AbstractVector{S2V},
              eupdates::AbstractVector{S3V})
    @unpack singles, ensembles = state
    step(wm, singles, ensembles, supdates, eupdates)
end

function step(wm::InertiaWM,
              singles::AbstractVector{InertiaSingle},
              ensembles::AbstractVector{InertiaEnsemble},
              supdates::AbstractVector{S2V},
              eupdates::AbstractVector{S3V})
    @unpack walls = wm
    ns = length(singles)
    new_singles = Vector{InertiaSingle}(undef, ns)
    @assert ns == length(supdates) "$(length(supdates)) updates but only $ns singles"
    ne = length(ensembles)
    new_ensembles = Vector{InertiaEnsemble}(undef, ne)
    @assert ne == length(eupdates) "$(length(eupdates)) updates but only $ne ensembles"
    @inbounds for i = 1:ns
        obj = singles[i]
        # force accumalator
        facc = MVector{2, Float64}(supdates[i])
        # interactions with walls
        for w in walls
            force!(facc, wm, w, obj)
        end
        new_singles[i] = update_state(obj, wm, facc)
    end
    @inbounds for i = 1:ne
        new_ensembles[i] = update_state(ensembles[i], wm,
                                        eupdates[i])
    end
    # println("step ns=$(length(new_singles))")
    InertiaState(PersistentVector(new_singles),
                 PersistentVector(new_ensembles))
end

"""Computes the force of A -> B"""
function force!(f::MVector{2, Float64}, ::InertiaWM, ::Object, ::Object)
    return nothing
end

function force!(f::MVector{2, Float64}, wm::InertiaWM, w::Wall, s::InertiaSingle)
    pos = get_pos(s)
    @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = wm
    n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing
end

# REVIEW: wall -> ensemble force
function force!(f::MVector{2, Float64}, wm::InertiaWM, w::Wall, e::InertiaEnsemble)
    # @unpack pos = d
    # @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    # n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    # f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing
end

"""
    update_state(::Object, ::GM, ::MVector{2, Float64})

resolve force on object
"""
function update_state end

function update_state(s::InertiaSingle, wm::InertiaWM, f::MVector{2, Float64})
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    @unpack mat, pos, vel, size = s
    @unpack area_height, area_width = wm
    # vx, vy = f
    vx, vy = vel + f
    vx = clamp(vx, -10., 10.)
    vy = clamp(vy, -10., 10.)
    x = clamp(pos[1] + vx,
              -area_width * 0.5,
              area_width * 0.5)
    y = clamp(pos[2] + vy,
              -area_height * 0.5,
              area_height * 0.5)
    new_pos = S2V(x, y)
    new_vel = S2V(vx, vy)
    # @show (vel, f, new_vel)
    InertiaSingle(mat, new_pos, new_vel, size)
end


function update_state(e::InertiaEnsemble, wm::InertiaWM, update::S3V)
    iszero(e.rate) && return e
    @unpack pos, vel, var = e
    f = S2V(update[1], update[2])
    bx, by = wm.dimensions
    # new_vel = dx, dy = f
    new_vel = dx, dy = vel + f
    x, y = pos
    new_pos = S2V(clamp(x + dx, -0.5 * bx, 0.5 * bx),
                  clamp(y + dy, -0.5 * by, 0.5 * by))
    new_var = var * update[3]
    # @show (var, update[3])
    setproperties(e; pos = new_pos,
                  vel = new_vel,
                  var = new_var)
end

"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    ns = length(singles)
    ne = length(ensembles)
    es = Vector{RandomFiniteElement{Detection}}(undef, ns + ne)
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    # add the single object representations
    @inbounds for i in 1:ns
        single = singles[i]
        pos = get_pos(single)
        args = (pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        # bw = inbounds(pos, bbmin, bbmax) ? 0.95 : 0.05
        es[i] = PoissonElement{Detection}(1.0, detect, args)
        # es[i] = IsoElement{Detection}(detect, args)
    end
    # the ensemble
    @inbounds for i = 1:ne
        @unpack matws, rate, pos, var = ensembles[i]
        mix_args = (matws,
                    [pos, pos],
                    [sqrt(var), sqrt(var)],
                    [1.0, 2.0],
                    [material_noise, material_noise])
        es[ns + i] =
            PoissonElement{Detection}(rate, detect_mixture, mix_args)
    end
    return es
end

# TODO: remove?
# function observe(gm::InertiaWM, singles::AbstractVector{InertiaSingle})
#     n = length(singles)
#     es = Vector{RandomFiniteElement{DetectionObs}}(undef, n)
#     @unpack single_noise, material_noise = wm
#     @inbounds for i in 1:n
#         single = singles[i]
#         args = (single.pos, single.size * single_noise, single.material,
#                 material_noise)
#         es[i] = IsoElement{Detection}(detect, args)
#     end
#     (es, gpp_mrfs(es, 200, 1.0))
# end

# include("helpers.jl")
include("gen.jl")

gen_fn(::InertiaWM) = wm_inertia
const InertiaIR = Gen.get_ir(wm_inertia)
const InertiaTrace = Gen.get_trace_type(wm_inertia)

function extract_rfs_subtrace(trace::InertiaTrace, t::Int64)
    # StaticIR names and nodes
    outer_ir = Gen.get_ir(wm_inertia)
    kernel_node = outer_ir.call_nodes[2] # :kernel
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    kernel_traces = getproperty(trace, kernel_field)
    sub_trace = kernel_traces.subtraces[t] # :kernel => t
    # StaticIR for `inertia_kernel`
    kernel_ir = Gen.get_ir(inertia_kernel)
    xs_node = kernel_ir.call_nodes[5] # :xs
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end


"""
    $(TYPEDSIGNATURES)

Posterior probability that gorilla is detected.
---
Criterion:
    1. A gorilla is present
    2. There are 5 singles
    3. All targets (xs[1-4]) are tracked
    4. The gorilla (x = n + 1) is tracked
"""
function detect_gorilla(trace::InertiaTrace,
                        ntargets::Int = 4,
                        nobj::Int = 8,
                        temp::Float64 = 1.0)

    t = first(get_args(trace))
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    scores = rfs.pscores
    nx,ne,np = size(pt)
    result = -Inf
    state = trace[:kernel => t]
    if (nx != nobj + 1 ) #|| length(state.singles) != 5)
        return result
    end
    @inbounds for p = 1:np
        targets_tracked = true
        for x = 1:ntargets
            pt[x, ne, p] || continue
            targets_tracked = false
            break
        end
        gorilla_detected = !pt[nx, ne, p]

        if targets_tracked && gorilla_detected
            result = logsumexp(result, scores[p] - rfs.score)
        end
    end
    return result
end

_kernel_prefix(t::Int, i::Int) = :kernel => t => :xs => i

# TODO: generalize observation size
function write_obs!(cm::ChoiceMap, wm::InertiaWM, positions,
                    t::Int,
                    gorilla_color::Material = Dark,
                    gorilla_idx::Int = 9;
                    single_size::Float64 = 10.0,
                    target_count::Int = 4,
                    prefix = _kernel_prefix)
    n = length(positions)
    for i = 1:n
        x, y = positions[i]
        mat = i <= target_count ? 1.0 : 2.0 # Light : Dark
        if i == gorilla_idx
            mat = gorilla_color == Light ? 1.0 : 2.0
        end
        cm[prefix(t, i)] = Detection(x, y, mat)
    end
    return nothing
end

function initial_state(wm::InertiaWM, positions, target_count::Int = 4)
    n = length(positions)
    singles = Vector{InertiaSingle}(undef, n)
    println("Loaded $(n) objects")
    for i = 1:n
        x, y = positions[i]
        color = i <= target_count ? Light : Dark
        # TODO: Check velocity
        singles[i] = InertiaSingle(color, S2V(x, y), S2V(0., 0.))
    end
    InertiaState(singles, InertiaEnsemble[])
end


function get_last_state(tr::InertiaTrace)
    t, wm, istate = get_args(tr)
    t == 0 ? istate : last(get_retval(tr))
end


function birth_weight(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensembles = st
    length(singles) + sum(rate, ensembles; init=0.0) <= wm.object_rate ?
        wm.birth_weight : 0.0
end

function add_baby_from_switch(prev, baby, idx)
    singles = PersistentVector(prev.singles)
    idx == 1 ?
        FunctionalCollections.push(singles, baby) :
        singles
end

function ensemble_var(wm::InertiaWM, spread::Float64)
    var = spread * wm.single_size^2
end

function split_merge_weights(wm::InertiaWM, x::InertiaState)
    ws = [2.0, 1.0, 1.0]
    @unpack singles, ensembles = x
    if isempty(ensembles)
        ws[2] = 0.0
    end
    if isempty(singles)
        ws[3] = 0.0
    end
    lmul!(1.0 / sum(ws), ws)
    return ws
end



include("visuals.jl")
# include("sm-kernel.jl")

function maybe_apply_split(st::InertiaState, baby, split::Bool)
    split || return st
    singles = add_baby_from_switch(st, baby, 1)
    ensemble = apply_split(st.ensemble, baby)
    InertiaState(singles, ensemble)
end

function apply_split(e::InertiaEnsemble, x::InertiaSingle)
    new_count = e.rate - 1
    matws = deepcopy(e.matws)
    lmul!(e.rate, matws)
    matws[Int64(x.mat)] -= 1
    lmul!(1.0 / new_count, matws)
    new_pos = (1 / new_count) .*
        (e.rate .* get_pos(e) - (1 / e.rate) .* get_pos(x))
    new_vel = (1 / new_count) .*
        (e.rate .* get_vel(e) - (1 / e.rate) .* get_vel(x))
    delta = sqrt(norm(get_pos(x) - get_pos(e)) *
        norm(get_pos(x) - new_pos))
    var = ((e.rate - 1) * e.var - delta) / (new_count - 1)
    var = max(100.0, var)
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function collapse(x::InertiaEnsemble)
    InertiaSingle(
        Material(argmax(x.matws)),
        get_pos(x),
        get_vel(x),
    )
end

function apply_splits(wm::InertiaWM, st::InertiaState, splits)
    all(isempty, splits) && return st
    @unpack singles, ensembles = st
    ne = length(ensembles)
    @assert length(ensembles) == length(splits)
    new_singles = Vector(singles)
    new_ensembles = InertiaEnsemble[]
    @inbounds for i = 1:ne
        ens = ensembles[i]
        if isempty(splits[i])
            push!(new_ensembles, ens)
            continue
        end
        for j = splits[i]
            ens = apply_split(ens, j)
            push!(new_singles, j)
        end

        if ens.rate > 1
            push!(new_ensembles, ens)
        else
            @assert ens.rate >= 0 "nonsensical ensemble rate $(ens.rate)"
            # ensemble -> individual
            # REVIEW: this should be the end of splits
            push!(new_singles, collapse(ens))
        end
    end
    println("applied splits")
    InertiaState(PersistentVector(new_singles),
                 PersistentVector(new_ensembles))
end

# TODO: parameterize with world model
function split_ppp(wm::InertiaWM, ensemble::InertiaEnsemble)
    y = sqrt(ensemble.var)
    ensemble.rate * max(0., 0.5 * log(y / 500.0))
end

const SplitRFS = RFGM(MRFS{InertiaSingle}(), (100, 1.0))

function object_from_idx(st::InertiaState, x::Int64)
    n = length(st.singles)
    x <= n ? st.singles[x] : st.ensembles[x - n]
end

function all_pairs(st::InertiaState)
    all_pairs(length(st.singles) + length(st.ensembles))
end

function all_pairs(n::Int64)
    nk = binomial(n, 2)
    xs = Vector{Int64}(undef, nk)
    ys = Vector{Int64}(undef, nk)
    c = 1
    @inbounds for i = 1:(n-1), j = (i+1):n
        xs[c] = i
        ys[c] = j
        c += 1
    end
    return (nk, xs, ys)
end

# TODO: Optimize for loop
function apply_mergers(st::InertiaState, mergers::AbstractVector{T}) where {T<:Bool}
    ens = st.ensemble
    n = min(length(st.singles), length(mergers))
    @inbounds for i = 1:n
        if mergers[i]
            ens = determ_merge(st.singles[i], ens)
        end
    end
    singles = PersistentVector(st.singles[.!(mergers)])
    InertiaState(singles, ens)
end

function get_maybe_add!(d::Dict, i::Int64, st::InertiaState)
    if haskey(d, i)
        return d[i]
    else
        o = d[i] = object_from_idx(st, i)
        return o
    end
end

function apply_mergers(st::InertiaState,
                       xs::Vector{Int64},
                       ys::Vector{Int64},
                       mergers::AbstractVector{T}) where {T<:Bool}
    iszero(count(mergers)) && return st
    n = length(xs)
    ns = length(st.singles)
    ne = length(st.ensembles)
    results = Dict{Int64, InertiaObject}()
    mapping = Dict{Int64, Int64}()
    remaining = trues(ns + ne)
    @inbounds for i = 1:n
        if mergers[i]
            # lookup indeces
            x = xs[i]; y = ys[i]
            remaining[x] = false
            remaining[y] = false
            # Int -> Int
            xi = get!(mapping, x, x)
            yi = get!(mapping, y, y)
            # println("joining $x -> $y ($(xi) -> $(yi))")
            if xi == yi
                # println("\t already merged")
                # already merged
                continue
            end
            # Retrieve objects Int -> Obj
            a = get_maybe_add!(results, xi, st)
            b = get_maybe_add!(results, yi, st)
            # println("\tx: $(typeof(a))\n\ty: $(typeof(b))")
            # remap
            delete!(results, xi)
            delete!(results, yi)
            mapping[x] = mapping[y] = xi
            # merge
            results[xi] = determ_merge(a, b)
        end
    end
    singles = PersistentVector(st.singles[remaining[1:ns]])
    new_ensembles = collect(InertiaEnsemble, values(results))
    ensembles = vcat(new_ensembles, st.ensembles[remaining[(ns+1):end]])
    # println("applied mergers")
    # @show (count(mergers), ns, ne)
    # @show (length(singles), sum(rate, ensembles))
    InertiaState(singles, ensembles)
end


function apply_merge(a::InertiaSingle, b::InertiaSingle)
    matws = zeros(NMAT)
    matws[Int64(a.mat)] += 1
    matws[Int64(b.mat)] += 1
    lmul!(0.5, matws)
    new_pos = 0.5 .* (get_pos(a) + get_pos(b))
    # REVIEW: dampen vel based on count?
    new_vel = 0.5 .* (get_vel(a) + get_vel(b))
    var = norm(get_pos(a) - get_pos(b))
    InertiaEnsemble(
        2,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function apply_merge(a::InertiaSingle, b::InertiaEnsemble)
    new_count = b.rate + 1
    matws = deepcopy(b.matws)
    lmul!(b.rate, matws)
    matws[Int64(a.mat)] += 1
    lmul!(1.0 / new_count, matws)
    new_pos = get_pos(b) + (get_pos(a) - get_pos(b)) / new_count
    # new_pos = (1 / new_count) .* get_pos(a) +
    #     (b.rate / new_count) .* get_pos(b)
    new_vel = (1 / new_count) .* get_vel(a) +
        (b.rate / new_count) .* get_vel(b)
    delta = sqrt(norm(get_pos(a) - get_pos(b)) *
        norm(get_pos(a) - new_pos))
    var = ((b.rate - 1) * b.var + delta) / (new_count - 1)
    # @show merge_probability(a, b)
    # @show a.pos
    # @show b.pos
    # @show delta
    # @show (b.var, var)
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

function apply_merge(a::InertiaEnsemble, b::InertiaSingle)
    apply_merge(b, a)
end

function apply_merge(a::InertiaEnsemble, b::InertiaEnsemble)
    new_count = a.rate + b.rate
    matws = a.rate .* a.matws + b.rate .* b.matws
    lmul!(1.0 / new_count, matws)
    new_pos = (a.rate / new_count) .* get_pos(a) +
        (b.rate / new_count) .* get_pos(b)
    new_vel = (a.rate / new_count) .* get_vel(a) +
        (b.rate / new_count) .* get_vel(b)
    var = a.var + b.var
    InertiaEnsemble(
        new_count,
        matws,
        new_pos,
        var,
        new_vel
    )
end

# TODO: parameterize with world model
function merge_probability(a::InertiaSingle, b::InertiaSingle)
    w = 0.05
    color = a.mat === b.mat ? 1.0 : 0.0
    l2 = norm(get_pos(a) - get_pos(b))
    ca = vec2_angle(get_vel(a), get_vel(b))
    color * (w * exp(-(l2)) + w * exp(-ca))
end

function merge_probability(a::InertiaSingle, b::InertiaEnsemble)
    w = 0.05
    iszero(b.rate) && return 0.0
    color = b.matws[Int64(a.mat)]
    l2 = norm(get_pos(a) - get_pos(b)) / sqrt(b.var)
    ca = vec2_angle(get_vel(a), get_vel(b))
    # @show l2
    # @show ca
    color * (w * exp(-(l2)) + w * exp(-ca))
end

function merge_probability(a::InertiaEnsemble, b::InertiaSingle)
    merge_probability(b, a)
end

function merge_probability(a::InertiaEnsemble, b::InertiaEnsemble)
    w = 0.05
    color = abs(a.matws[1] - b.matws[1]) # HACK
    l2 = norm(get_pos(a) - get_pos(b)) / (sqrt(b.var + a.var))
    ca = vec2_angle(get_vel(a), get_vel(b))
    color * (w * exp(-(l2)) + w * exp(-ca))
end
