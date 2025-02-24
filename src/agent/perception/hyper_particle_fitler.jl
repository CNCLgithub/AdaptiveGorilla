abstract type PerceptionModule end

struct HyperFilter{P<:AbstractParticleFilter
                   } <: PerceptionModule
    "Number of hyper particle chains"
    h::Int
    "Number of particles (per chain)"
    n::Int
    "Number of observations per epoch"
    dt::Int
    "Particle filter procedure"
    pf::P
end

function perceive!(chain, pm::HyperFilter)

end
