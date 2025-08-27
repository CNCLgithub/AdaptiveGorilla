using Statistics
using Distributions


normal_dist = Normal(0., 1.)

function ensemble_prob(mu::Float64, var::Float64,
                       rate::Float64, d::Float64)
    z = (d - mu) / sqrt(var)
    pnocol = cdf(normal_dist, z)
    return 1.0 - exp(rate * log(pnocol))
end


const NEGLN2 = -log(2)

"""
Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
See [Maechler2012accurate] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
"""
function log1mexp(x::Float64)
    x = min(x, 0.0)
    x  > NEGLN2 ? log(-expm1(x)) : log1p(-exp(x))
end


function ensemble_prob_stable(mu::Float64, var::Float64,
                       rate::Float64, d::Float64)
    z = (d - mu) / sqrt(var)
    logprobnocol = logcdf(normal_dist, z)
    @show logprobnocol
    return log1mexp(rate * logprobnocol)
    # logp = log1mexp(logcdf(normal_dist, z))
    # @show logp
    # @show -log(rate) + logp
    # return log1mexp(-log(rate) + logp)
end

function test()
    d = 2.0
    var = 1.0
    rate = 3.0
    prob = ensemble_prob(0., var, rate, d)
    logprob = ensemble_prob_stable(0., var, rate, d)
    println("d=$d, var=$var, prob=$prob")
    println("d=$d, var=$var, logprob=$(log(prob))")
    println("d=$d, var=$var, logprob=$(logprob)")

    d = 100.0
    var = 45.0
    rate = 4.0
    prob = ensemble_prob(0., var, rate, d)
    logprob = ensemble_prob_stable(0., var, rate, d)
    println("d=$d, var=$var, prob=$prob")
    println("d=$d, var=$var, logprob=$(logprob)")
end

test();
