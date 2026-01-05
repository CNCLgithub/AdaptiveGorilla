export CIsoElement

"""
$(TYPEDEF)

Numerically stable Bernoulli element with low miss chance.
"""
@kwdef struct CIsoElement{T} <: GenRFS.EpimorphicRFE{T}
    d::Gen.Distribution{T}
    args::Tuple
    log_penalty::Float64 = -3000.0
end

GenRFS.distribution(rfe::CIsoElement) = rfe.d
GenRFS.args(rfe::CIsoElement) = rfe.args

const cpoisson_cardinality = truncated(Poisson(1.1); upper = 2)

function GenRFS.cardinality(rfe::CIsoElement, n::Int)
    n == 1 ? 0.0 : rfe.log_penalty
end

function GenRFS.sample_cardinality(rfe::CIsoElement)
    1 # HACK
end
