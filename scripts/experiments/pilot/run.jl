using Gen
using ArgParse
using Gen_Compose
using DataFrames, CSV
using AdaptiveGorilla

att_modules = Dict(
    :ac => AdaptiveComputation,
    :un => UniformProtocol
)

att_configs = Dict(
    :ac => "$(@__DIR__)/ac.json",
    :un => "$(@__DIR__)/un.json",
)

function gorilla_exp(logger)
    bfr = buffer(logger)
    n = length(bfr)
    pgorilla = 0.0
    for i = 1:n
        pgorilla += bfr[i][:pgorilla]
    end
    pgorilla /= n
    return pgorilla
end

function main()
    trial = 2
    duration = 30
    att_module = :ac
    dataset = "pilot"
    gorilla_color = Light
    chains = 1

    result = DataFrame(:trial => Int64[],
                       :color => Symbol[],
                       :chain => Int64[],
                       :pgorilla => Float64[])

    # wm_config = load_json("$(@__DIR__)/wm.json")
    # wm = InertiaWM(;wm_config...)
    wm = InertiaWM(area_width = 1240.0,
                   area_height = 840.0,
                   birth_weight = 0.1,
                   single_noise = 0.5,
                   stability = 0.90,
                   vel = 3.0,
                   force_low = 1.0,
                   force_high = 5.0,
                   material_noise = 0.001,
                   ensemble_shape = 1.2,
                   ensemble_scale = 1.0,
                   wall_rep_m = 0.0)
    display(wm)
    dpath = "/spaths/datasets/$(dataset).json"
    query = query_from_dataset(wm, dpath, trial, duration, gorilla_color)

    att_config = load_json(att_configs[att_module])
    att = att_modules[att_module](;att_config...)
    proc = AdaptiveParticleFilter(;load_json("$(@__DIR__)/pf.json")...,
                                  attention = att)
    nsteps = length(query)

    chain = 1
    logger = MemLogger(nsteps)
    run_chain(proc, query, nsteps, logger)
    pgorilla = gorilla_exp(logger)
    push!(result, (trial, Symbol(gorilla_color), chain, pgorilla))
    # for chain = 1:chains
    #     logger = MemLogger(nsteps)
    #     run_chain(proc, query, nsteps, logger)
    #     pgorilla = gorilla_exp(logger)
    #     push!(result, (trial, Symbol(gorilla_color), chain, pgorilla))
    # end

    display(result)

    out = "/spaths/experiments/$(dataset)/$(trial)_$(gorilla_color)"
    isdir(out) || mkpath(out)
    CSV.write("$(out).csv", result)
    render_inference(wm, logger, out)

end;


main();
