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
    singles::AbstractVector{InertiaSingle}
    ensemble::InertiaEnsemble
end

function InertiaState(wm::InertiaWM,
                      singles::AbstractVector{InertiaSingle},
                      ensemble::InertiaEnsemble)
    InertiaState(singles, ensemble)
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
     force_low,
     force_high)
end


function step(wm::InertiaWM,
              state::InertiaState,
              updates::AbstractVector{S2V},
              eupdate::S3V)
    @unpack singles, ensemble = state
    step(wm, singles, ensemble, updates, eupdate)
end

function step(wm::InertiaWM,
              singles::AbstractVector{InertiaSingle},
              ensemble::InertiaEnsemble,
              updates::AbstractVector{S2V},
              eupdate::S3V)
    @unpack walls = wm
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
    InertiaState(PersistentVector(new_singles), new_ensemble)
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
    vx, vy = vel + f
    vx = clamp(vx, -10., 10.)
    vy = clamp(vy, -10., 10.)
    x = pos[1] + vx
    y = pos[2] + vy
    x = clamp(x, -area_width * 0.5,
              area_width * 0.5)
    y = clamp(y, -area_height * 0.5,
              area_height * 0.5)
    new_pos = S2V(x, y)
    new_vel = S2V(vx, vy)
    InertiaSingle(mat, new_pos, new_vel, size)
end


function update_state(e::InertiaEnsemble, wm::InertiaWM, update::S3V)
    @unpack pos, vel, var = e
    f = S2V(update[1], update[2])
    bx, by = wm.dimensions
    new_vel = dx, dy = vel + f
    x, y = pos
    new_pos = S2V(clamp(x + dx, -0.5 * bx, 0.5 * bx),
                  clamp(y + dy, -0.5 * by, 0.5 * by))
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
    @unpack single_noise, material_noise, bbmin, bbmax = wm
    # add the single object representations
    @inbounds for i in 1:n
        single = singles[i]
        pos = get_pos(single)
        args = (pos, single.size * single_noise,
                Float64(Int(single.mat)),
                material_noise)
        # bw = inbounds(pos, bbmin, bbmax) ? 0.95 : 0.05
        # es[i] = BernoulliElement{Detection}(bw, detect, args)
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
    # @show length(singles)
    # @show rate
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
    xs_node = kernel_ir.call_nodes[4] # :xs
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

# TODO: generalize observation size
function write_obs!(cm::ChoiceMap, wm::InertiaWM, positions,
                    t::Int,
                    gorilla_color::Material = Dark,
                    gorilla_idx::Int = 9,
                    single_size::Float64 = 10.0,
                    target_count::Int = 4)
    n = length(positions)
    for i = 1:n
        x, y = positions[i]
        mat = i <= target_count ? 0.0 : 1.0 # Light : Dark
        if i == gorilla_idx
            mat = gorilla_color == Light ? 0.0 : 1.0
        end
        cm[:kernel => t => :xs => i] = Detection(x, y, mat)
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


function add_baby_from_switch(prev, baby, idx)
    idx == 1 ?
        FunctionalCollections.push(prev.singles, baby) :
        prev.singles
end

include("visuals.jl")
