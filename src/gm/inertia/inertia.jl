export InertiaWM, InertiaState

#REVIEW: why override this method?
# import MOTCore: get_pos

################################################################################
# Model definition
################################################################################

"""
$(TYPEDEF)

Model that uses inertial change points to "explain" interactions

# Fields:

$(TYPEDFIELDS)
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
    single_rfs_logweight::Float64 = -2500.0
    # Ensemble parameters
    # REVIEW: Are these used?
    # ensemble_shape::Float64 = 2.0
    # ensemble_scale::Float64 = 0.5

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


################################################################################
# Object Representations
################################################################################

"Object representations for `InertiaWM`"
abstract type InertiaObject <: Object end

"""
$(TYPEDEF)

An individual object representation for `InertiaWM`

# Fields:

$(TYPEDFIELDS)
"""
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
get_size(s::InertiaSingle) = s.size
material(s::InertiaSingle) = s.mat

"""
$(TYPEDEF)

An ensemble object representation for `InertiaWM`.
Acts like a [Poission-point-process](https://en.wikipedia.org/wiki/Poisson_point_process)

# Fields:

$(TYPEDFIELDS)
"""
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
get_var(e::InertiaEnsemble) = e.var
rate(e::InertiaEnsemble) = e.rate
materials(e::InertiaEnsemble) = e.matws

################################################################################
# State
################################################################################

"""
$(TYPEDEF)

State for `InertiaWM`.

#Fields:

$(TYPEDFIELDS)
"""
struct InertiaState <: WorldState{InertiaWM}
    singles::AbstractVector{InertiaSingle}
    ensembles::AbstractVector{InertiaEnsemble}
end

function InertiaState(wm::InertiaWM,
                      singles::AbstractVector{InertiaSingle})
    InertiaState(singles, InertiaEnsemble[])
end

function InertiaState(wm::InertiaWM,
                      singles::AbstractVector{InertiaSingle},
                      ensembles::AbstractVector{InertiaEnsemble})
    InertiaState(singles, ensembles)
end

function object_from_idx(st::InertiaState, x::Int64)
    n = length(st.singles)
    x <= n ? st.singles[x] : st.ensembles[x - n]
end

function object_count(st::InertiaState)
    n = Float64(length(st.singles))
    sum(rate, st.ensembles; init=n)
end

################################################################################
# Components
################################################################################

include("dynamics.jl")   # Motion, Birth-death
include("graphics.jl")   # Likelihood, RFS
include("splitmerge.jl") # Granularity shifts
include("gen.jl")        # Conditional probabilities
include("trace.jl")      # Trace methods (RFS intermediate, retval)
include("io.jl")         # Trial loading
include("visuals.jl")    # Visualizations
