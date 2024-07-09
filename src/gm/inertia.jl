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

    # Ensemble parameters
    ensemble_shape::Float64 = 1.0
    ensemble_scale::Float64 = 1.0

    "Probability that a baby is born for a given step"
    birth_weight::Float64 = 0.01


    # DYNAMICS
    # - general
    vel::Float64 = 10 # base vel
    area_width::Float64 = 400.0
    area_height::Float64 = 400.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)
    display_border::Float64 = 20.0
    bbmin::S2V = S2V(area_width + display_border,
                     area_height + display_border)
    bbmax::S2V = S2V(area_width - display_border,
                     area_height - display_border)

    # - inertia_force
    "Probability that an individual is stable for a given step"
    stability::Float64 = 0.9
    force_low::Float64 = 0.1
    force_high::Float64 = force_low * 10.0
    ensemble_var_shift::Float64 = 0.3

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

struct InertiaState <: WorldState{InertiaWM}
    walls::SVector{4, Wall}
    singles::AbstractVector{InertiaSingle}
    ensemble::InertiaEnsemble
end

function InertiaState(wm::InertiaWM,
                        singles::AbstractVector{InertiaSingle},
                        ensemble::InertiaEnsemble)
    walls = MOTCore.init_walls(wm.area_width)
    InertiaState(walls, singles, ensemble)
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
    (exp(log(rate) * log(stability)),
     force_low / rate,
     force_high / rate)
end


function step(wm::InertiaWM,
              state::InertiaState,
              updates::AbstractVector{S2V},
              eupdate::S3V)
    @unpack walls, singles, ensemble = state
    step(wm, singles, ensemble, walls, updates, eupdate)
end

function step(wm::InertiaWM,
              singles::AbstractVector{InertiaSingle},
              ensemble::InertiaEnsemble,
              walls::AbstractVector{Wall},
              updates::AbstractVector{S2V},
              eupdate::S3V)

    n = length(singles)
    new_singles = Vector{InertiaSingle}(undef, n)

    @assert n == length(updates) "$(length(updates)) updates but only $n singles"

    @inbounds for i = 1:n
        obj = singles[i]
        # force accumalator
        facc = MVector{2, Float64}(updates[i])
        # interactions with walls
        for w in walls
            force!(facc, wm, w, obj)
        end
        new_singles[i] = update_state(obj, wm, facc)
    end

    new_ensemble = update_state(ensemble, wm, eupdate)

    InertiaState(walls, singles, ensemble)
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
    new_vel = vel + f
    x, y = pos + new_vel
    x = clamp(x, -area_width * 0.5 + size,
              area_width * 0.5 - size)
    y = clamp(x, -area_height * 0.5 + size,
              area_height * 0.5 - size)
    new_pos = S2V(x, y)
    InertiaSingle(mat, new_pos, new_vel, size)
end


function update_state(e::InertiaEnsemble, wm::InertiaWM, update::S3V)
    @unpack pos, vel, var = e
    f = S2V(update[1], update[2])
    bx, by = wm.dimensions
    new_vel = dx, dy = vel + f
    x, y = pos
    new_pos = S2V(clamp(x + dx, -bx, bx),
                  clamp(y + dy, -by, by))
    new_var = var * update[3]
    setproperties(e; pos = new_pos,
                  vel = new_vel,
                  var = new_var)
end

"""
$(TYPEDSIGNATURES)

The random finite elements corresponding to the world state.
"""
function predict(wm::InertiaWM, st::InertiaState)
    @unpack singles, ensemble = st
    n = length(singles)
    es = Vector{RandomFiniteElement{Detection}}(undef, n + 1)
    @unpack single_noise, material_noise = wm
    # add the single object representations
    @inbounds for i in 1:n
        single = singles[i]
        args = (single.pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        es[i] = IsoElement{Detection}(detect, args)
    end
    # the ensemble
    @unpack matws, rate, pos, var = ensemble
    mix_args = (matws,
                [pos, pos],
                [var, var],
                [0.0, 1.0],
                [material_noise, material_noise])
    es[n + 1] = PoissonElement{Detection}(rate, detect_mixture, mix_args)
    @show length(singles)
    @show rate
    return es
end

function observe(gm::InertiaWM, singles::AbstractVector{InertiaSingle})
    n = length(singles)
    es = Vector{RandomFiniteElement{DetectionObs}}(undef, n)
    @unpack single_noise, material_noise = wm
    @inbounds for i in 1:n
        single = singles[i]
        args = (single.pos, single.size * single_noise, single.material,
                material_noise)
        es[i] = IsoElement{Detection}(detect, args)
    end
    (es, gpp_mrfs(es, 200, 1.0))
end

# include("helpers.jl")
include("inertia_gen.jl")

gen_fn(::InertiaWM) = wm_inertia
const InertiaIR = Gen.get_ir(wm_inertia)
const InertiaTrace = Gen.get_trace_type(wm_inertia)

function extract_rfs_subtrace(trace::InertiaTrace, t::Int64)
    # StaticIR names and nodes
    outer_ir = Gen.get_ir(gm_inertia)
    kernel_node = outer_ir.call_nodes[2] # (:kernel)
    kernel_field = Gen.get_subtrace_fieldname(kernel_node)
    # subtrace for each time step
    vector_trace = getproperty(trace, kernel_field)
    sub_trace = vector_trace.subtraces[t]
    # StaticIR for `inertia_kernel`
    inner_ir = Gen.get_ir(inertia_kernel)
    xs_node = inner_ir.call_nodes[2] # (:masks)
    xs_field = Gen.get_subtrace_fieldname(xs_node)
    # `RFSTrace` for :masks
    getproperty(sub_trace, xs_field)
end

function td_flat(trace::InertiaTrace, temp::Float64)

    t = first(get_args(trace))
    rfs = extract_rfs_subtrace(trace, t)
    pt = rfs.ptensor
    # @unpack pt, pls = st
    nx,ne,np = size(pt)
    ne -= 1
    # ls::Float64 = logsumexp(pls)
    nls = log.(softmax(rfs.pscores, t=temp))
    # probability that each observation
    # is explained by a target
    x_weights = Vector{Float64}(undef, nx)
    @inbounds for x = 1:nx
        xw = -Inf
        @views for p = 1:np
            if !pt[x, ne + 1, p]
                xw = logsumexp(xw, nls[p])
            end
        end
        # @views for p = 1:np, e = 1:ne
        #     pt[x, e, p] || continue
        #     xw = logsumexp(xw, nls[p])
        # end
        x_weights[x] = xw
    end

    # @show length(pls)
    # display(sum(pt; dims = 3))
    # @show x_weights
    # the ratio of observations explained by each target
    # weighted by the probability that the observation is
    # explained by other targets
    td_weights = fill(-Inf, ne)
    @inbounds for i = 1:ne
        for p = 1:np
            ew = -Inf
            @views for x = 1:nx
                pt[x, i, p] || continue
                ew = x_weights[x]
                # assuming isomorphicity
                # (one association per partition)
                break
            end
            # P(e -> x) where x is associated with any other targets
            prop = nls[p]
            ew += prop
            td_weights[i] = logsumexp(td_weights[i], ew)
        end
    end
    return td_weights
end

# TODO: generalize observation size
function write_obs!(cm::ChoiceMap, wm::InertiaWM, positions,
                    t::Int,
                    single_size::Float64 = 10.0,
                    target_count::Int = 4)
    n = length(positions)
    es = Vector{RandomFiniteElement{Detection}}(undef, n)
    @unpack single_noise, material_noise = wm
    @inbounds for i in 1:n
        pos = S2V(positions[i])
        mat = i <= target_count ? 0.0 : 1.0 # Light : Dark
        args = (pos, single_size * single_noise, mat,
                material_noise)
        es[i] = IsoElement{Detection}(detect, args)
    end
    xs = DetectionRFS(es)
    for (i, x) = enumerate(xs)
        cm[:kernel => t => :xs => i] = x
    end
    return nothing
end

function write_initial_constraints!(cm::ChoiceMap, wm::InertiaWM, positions,
                                    target_count = 4)
    n = length(positions)
    # prior over object partitioning
    cm[:init_state => :n] = n
    cm[:init_state => :k] = target_count
    # prior over singles
    for i = 1:target_count
        x, y = positions[i]
        cm[:init_state => :singles => i => :material] = 1 # Light
        cm[:init_state => :singles => i => :state => :x] = x
        cm[:init_state => :singles => i => :state => :y] = y
    end
    # prior over ensemble
    ex, ey = mean(positions[(target_count+1):n])
    cm[:init_state => :ensemble => :plight] = 0.01
    cm[:init_state => :ensemble => :state => :x] = ex
    cm[:init_state => :ensemble => :state => :y] = ey
    return nothing
end
