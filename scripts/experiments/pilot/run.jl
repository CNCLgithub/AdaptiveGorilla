using Gen
using ArgParse
using Gen_Compose
using DataFrames, CSV
using AdaptiveGorilla

att_modules = Dict(
    :ac => AdaptiveProtocol,
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
    trial = 1
    duration = 40
    att_module = :ac
    dataset = "pilot"
    gorilla_color = Light
    chains = 5

    result = DataFrame(:trial => Int64[],
                       :color => Symbol[],
                       :chain => Int64[],
                       :pgorilla => Float64[])

    wm_config = load_json("$(@__DIR__)/wm.json")
    wm = InertiaWM(;wm_config...)
    display(wm)
    dpath = "output/datasets/$(dataset).json"
    query = query_from_dataset(wm, dpath, trial, duration, gorilla_color)

    att_config = load_json(att_configs[att_module])
    att = att_modules[att_module](;att_config...)
    proc = AdaptiveParticleFilter(;load_json("$(@__DIR__)/pf.json")...,
                                  attention = att)
    nsteps = length(query)
    for chain = 1:chains
        logger = MemLogger(nsteps)
        run_chain(proc, query, nsteps, logger)
        pgorilla = gorilla_exp(logger)
        push!(result, (trial, Symbol(gorilla_color), chain, pgorilla))
    end

    display(result)

    out = "output/experiments/$(dataset)/$(trial)_$(gorilla_color)"
    isdir(out) || mkpath(out)
    CSV.write("$(out).csv", result)
    # render_inference(wm, logger, out)

end;


main();
