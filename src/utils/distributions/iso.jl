export IsoElement

struct IsoElement{T} <: GenRFS.IsomorphicRFE{T}
    d::Gen.Distribution{T}
    args::Tuple
end

GenRFS.distribution(rfe::IsoElement) = rfe.d
GenRFS.args(rfe::IsoElement) = rfe.args

function GenRFS.cardinality(::IsoElement, n::Int)
    n === 1 ? 0. : -Inf
end

function GenRFS.sample_cardinality(rfe::IsoElement)
    1
end
