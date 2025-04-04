export NegBinomElement

import GenRFS

struct NegBinomElement{T} <: GenRFS.EpimorphicRFE{T}
    r::Float64
    p::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

GenRFS.distribution(rfe::NegBinomElement) = rfe.d
GenRFS.args(rfe::NegBinomElement) = rfe.args

GenRFS.cardinality(rfe::NegBinomElement, n::Int) =
    n < 0 ? -Inf : Distributions.logpdf(NegativeBinomial(rfe.r, rfe.p), n)

GenRFS.sample_cardinality(rfe::NegBinomElement) =
    Distributions.rand(NegativeBinomial(rfe.r, rfe.p))
