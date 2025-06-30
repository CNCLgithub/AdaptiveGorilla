export LogBernElement

"""
$(TYPEDEF)

Numerically stable Bernoulli element with low miss chance.
"""
struct LogBernElement{T} <: GenRFS.MonomorphicRFE{T}
    "Log-weight of x = {}"
    logw::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

GenRFS.distribution(rfe::LogBernElement) = rfe.d
GenRFS.args(rfe::LogBernElement) = rfe.args

function GenRFS.cardinality(rfe::LogBernElement, n::Int)
    n > 1 && return -Inf
    n == 1 ?  log1mexp(rfe.logw) : rfe.logw
end

function GenRFS.sample_cardinality(rfe::LogBernElement)
    Int64(log(rand()) > rfe.logw)
end
