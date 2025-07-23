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

function GenRFS.cardinality(rfe::NegBinomElement, n::Int)
    Gen.logpdf(binom, n, 9, 0.13)
    # n < 0 ? -Inf : Distributions.logpdf(dnp, n)
end

GenRFS.sample_cardinality(rfe::NegBinomElement) =
    binom(9, 0.13)
