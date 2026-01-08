export CPoissonElement

"""
$(TYPEDEF)

Numerically stable Bernoulli element with low miss chance.
"""
@kwdef struct CPoissonElement{T} <: GenRFS.EpimorphicRFE{T}
    rate::Float64
    d::Gen.Distribution{T}
    args::Tuple
    log_penalty::Float64 = -3000.0
end

GenRFS.distribution(rfe::CPoissonElement) = rfe.d
GenRFS.args(rfe::CPoissonElement) = rfe.args

function GenRFS.cardinality(rfe::CPoissonElement, n::Int)
    rv = truncated(Distributions.Poisson(rfe.rate); lower = 1)
    correction = log1mexp(Gen.logpdf(poisson, 0, rfe.rate))
    n > 1 ?
        Gen.logpdf(poisson, n, rfe.rate) - correction :
        rfe.log_penalty
end

function GenRFS.sample_cardinality(rfe::CPoissonElement)
    rand(truncated(Poisson(rfe.rate); lower = 1))
end
