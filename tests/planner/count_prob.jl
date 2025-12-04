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

# test();


using AdaptiveGorilla: InertiaSingle, init_walls, closest_wall, Light, colprob_and_agrad

function test_case_1()
    s = InertiaSingle(Light,
[12.457043041353916, 89.90576639138374],
# [9.0, -4.372010699572394],
[9.0, 9.0],
                      10.0)
    walls = init_walls(720.0, 480.0)
    closest = walls[closest_wall(s, walls)]
    (colprob, dpi) = colprob_and_agrad(s, closest)
    @show colprob
    @show dpi
end

test_case_1();


# tr = [-31.977765833136836, -102.04198738809453, -95.74388636442394, -Inf, -6377.234786640864]
# pos: [-18.59464849834508, 103.45410724065383]
#  vel: [-8.445741990158425, -2.1493133697897058]
# tr = [-27.27965866924172, -100.18067973524569, -123.088800871827, -Inf, -6375.409967303469]
# pos: [-11.300671230328982, 112.46275568069179]
#  vel: [-1.1517647221423264, 6.859335070248246]
# tr = [-35.752817510411546, -101.77864225528599, -123.00532994719465, -Inf, -6375.9406891979925]
# pos: [-19.148906508186656, 96.60342061044354]
#  vel: [-9.0, -9.0]
# tr = [-27.689493452478672, -97.71996259022167, -122.09204755924533, -Inf, -6379.280731971207]
# pos: [-7.45097645905469, 110.0433383184319]
#  vel: [2.9475417859343374, -7.179412620222553]
# tr = [-26.212811127749603, -89.12866569688909, -123.04015416222765, -Inf, -6376.746499484055]
# pos: [8.154498744542995, 104.7024129958036]
#  vel: [5.894495507629273, -6.584124877069231]
