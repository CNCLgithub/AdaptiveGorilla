export InertiaWM, InertiaState

################################################################################
# Model definition
################################################################################

"""
Model that uses inertial change points to "explain" interactions
"""
@with_kw struct InertiaWM <: WorldModel

    # EPISTEMICS
    "Possible materials for objects. Sampled uniformly"
    materials::Vector{Material} = instances(Material)
    "Average number of objects in the scene"
    object_rate::Float64 = 8.0
    "Number of individual representations"
    irate::Float64 = 0.5

    # Ensemble parameters
    ensemble_shape::Float64
    ensemble_scale::Float64

    "Probability that a baby is born for a given step"
    birth_weight::Float64 = 0.01


    # DYNAMICS
    # - general
    vel::Float64 = 10 # base vel
    dot_radius::Float64 = 20.0
    area_width::Float64 = 400.0
    area_height::Float64 = 400.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)

    # - inertia_force
    "Probability that an individual is stable for a given step"
    istability::Float64 = 0.9
    ang_var::Float64 = 100.0
    mag_var::Float64 = 2.5

    # - wall force
    wall_rep_m::Float64 = 0.0
    wall_rep_a::Float64 = 0.02
    wall_rep_x0::Float64 = 0.0
end

materials(wm::InertiaWM) = wm.materials

abstract type InertiaObject end

@with_kw struct InertiaSingle <: InertiaObject
    mat::Material
    pos::S2V
    vel::S2V
    size::Float64 = 10.0
end

struct InertiaEnsemble <: InertiaObject
    rate::Float64
    mat::Material
    pos::S2V
    var::Float64
    vel::S2V
end

struct InertiaState <: WorldState{InertiaWM}
    walls::SVector{4, Wall}
    singles::Vector{InertiaSingle}
    ensemble::InertiaEnsemble
end

function step(wm::InertiaWM,
              state::InertiaState,
              updates::AbstractVector{S2V})

    # Dynamics (computing forces)
    # for each dot compute forces
    @unpack walls, singles, ensemble = state
    ns = length(singles)
    next = Vector{InertiaSingle}(undef, ns)

    @inbounds for i = 1:ns
        # force accumalator
        s = singles[i]
        # initialized from the motion kernel
        facc = MVector{2, Float64}(updates[i])
        # interactions with walls
        for w in walls
            force!(facc, wm, w, s)
        end
        next[i] = update_state(s, wm, facc)
    end
    return next
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

# TODO
function force!(f::MVector{2, Float64}, dm::InertiaWM, w::Wall, e::InertiaEnsemble)
    # @unpack pos = d
    # @unpack wall_rep_m, wall_rep_a, wall_rep_x0 = dm
    # n = LinearAlgebra.norm(w.normal .* pos + w.nd)
    # f .+= wall_rep_m * exp(-1 * (wall_rep_a * (n - wall_rep_x0))) * w.normal
    return nothing
end

"""
    update_state(::GM, ::Object, ::MVector{2, Float64})

resolve force on object
"""
function update_state end

function update_state(wm::InertiaWM, s::InertiaSingle, f::MVector{2, Float64})
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    @unpack pos, vel, size = s
    @unpack area_height = wm
    new_vel = vel + f
    new_pos = clamp.(pos + new_vel,
                     -area_height * 0.5 + d.radius,
                     area_height * 0.5  - d.radius)
    KinematicsUpdate(new_pos, new_vel)

"""
    update_state(::WM, ::Object, ::MVector{2, Float64})

resolve force on object, returning state update
"""
function update_state end

function update_state(wm::InertiaWM, d::Dot, f::MVector{2, Float64})
    # treating force directly as velocity;
    # update velocity by x percentage;
    # but f isn't normalized to be similar to v
    a = f/d.mass
    new_vel = d.vel + a
    new_pos = clamp.(get_pos(d) + new_vel,
                     -wm.area_height * 0.5 + d.radius,
                     wm.area_height * 0.5  - d.radius)
end


function predict(gm::InertiaWM,
                 t::Int,
                 st::InertiaState,
                 objects::AbstractVector{Dot})
    n = length(objects)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n + 1)
    # es = RFSElements{GaussObs{2}}(undef, n + 1)
    # the trackers
    @unpack area_width, k_tail, tail_sample_rate = gm
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp,
                                       (obj.gstate,))
    end
    # the ensemble
    tback = t < k_tail ? (t + 1) : k_tail
    nt = ceil(Int64, tback / tail_sample_rate)
    w = -log(nt) # REVIEW: no longer used in `GaussianComponent`
    @unpack rate = (st.ensemble)
    mu = @SVector zeros(2)
    cov = SMatrix{2,2}(spdiagm([50*area_width, 50*area_width]))
    uniform_gpp = Fill(GaussianComponent{2}(w, mu, cov), nt)
    es[n + 1] = PoissonElement{GaussObs{2}}(rate, gpp, (uniform_gpp,))
    return es
end

function observe(gm::InertiaWM,
                 objects::AbstractVector{Dot})
    n = length(objects)
    # es = RFSElements{GaussObs{2}}(undef, n)
    es = Vector{RandomFiniteElement{GaussObs{2}}}(undef, n)
    @inbounds for i in 1:n
        obj = objects[i]
        es[i] = IsoElement{GaussObs{2}}(gpp, (obj.gstate,))
    end
    (es, gpp_mrfs(es, 50, 1.0))
end

include("helpers.jl")
include("gen.jl")

gen_fn(::InertiaWM) = gm_inertia
const InertiaIr = Gen.get_ir(gm_inertia)
const InertiaTrace = Gen.get_trace_type(gm_inertia)

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
