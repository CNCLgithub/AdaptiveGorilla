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

# pos: [-59.83589127805614, -91.11910009764436]
#  vel: [-13.5, 0.01556850670921861]
# closest = MOTCore.Wall(240.00000000000006, [-1.8369701987210297e-16, -1.0], [-4.408728476930472e-14, -240.00000000000006])
# _colprob = -0.0
# colprob = 1.147970607462346e-13

# x = [94.08386704821129, -36.87527629129443]
# v = [-9.0, -2.7831946253101987]
# v_orth = 2.7831946253102005
# dt = 71.18608303816471
# sigma = 1.5346439885590686
# z = -45.73443975372093
# dpdz = -1046.7384283265692

function test_case_1()

    walls = init_walls(720.0, 480.0)

    s = InertiaSingle(Light,
                      [-70.4895197369258, 169.91693767757113], [3.7188411340994745, 6.366964648542833], 0.05,
                      )
    closest = walls[closest_wall(s, walls)]
    @show s
    @show closest
    (colprob, dpi) = colprob_and_agrad(s, closest)
    @show colprob
    @show dpi

    s = InertiaSingle(Light,
                      [39.36612868392137, 230.0],
                      [2.2135894220916565, 4.5],
                      -0.3,
                      )
    closest = walls[closest_wall(s, walls)]
    @show s
    @show closest
    (colprob, dpi) = colprob_and_agrad(s, closest)
    @show colprob
    @show dpi

    # s = InertiaSingle(Light,

    #      [12.0, 210.],
    #                   [9.0, 9.0])
    # closest = walls[closest_wall(s, walls)]
    # @show s
    # @show closest
    # (colprob, dpi) = colprob_and_agrad(s, closest)
    # @show colprob
    # @show dpi


    # s = InertiaSingle(Light,
    #                   [12.0, 235.],
    #                   [9.0, 9.0])
    # closest = walls[closest_wall(s, walls)]
    # @show s
    # @show closest
    # (colprob, dpi) = colprob_and_agrad(s, closest)
    # @show colprob
    # @show dpi
end

test_case_1();

using AdaptiveGorilla: InertiaEnsemble

function test_case_2()
    walls = init_walls(720.0, 480.0)
    e = InertiaEnsemble(3.0,
                        [1.0, 0.0],
                        [20., 200.0],
                        150.0,
                        [-5.0, 10.0])
    closest = walls[closest_wall(e, walls)]
    @show e
    @show closest
    (colprob, dpi) = colprob_and_agrad(e, closest)
    @show colprob
    @show dpi
end

# test_case_2();
