export CPoissonElement

"""
$(TYPEDEF)

Numerically stable Bernoulli element with low miss chance.
"""
struct CPoissonElement{T} <: GenRFS.EpimorphicRFE{T}
    d::Gen.Distribution{T}
    args::Tuple
end

GenRFS.distribution(rfe::CPoissonElement) = rfe.d
GenRFS.args(rfe::CPoissonElement) = rfe.args

const cpoisson_cardinality = truncated(Poisson(1.1); upper = 2)

function GenRFS.cardinality(rfe::CPoissonElement, n::Int)
    n == 1 ? 0.0 : -3000.0 # HACK
end

function GenRFS.sample_cardinality(rfe::CPoissonElement)
    1 # HACK
end
