using Gen
using CSV
using JSON3
using MOTCore
using FillArrays
using StaticArrays
using UnicodePlots
using Statistics: mean
using LinearAlgebra: norm, cross, dot
using AdaptiveGorilla: ncr, S2V
using Accessors: setproperties
using MOTCore: scholl_delta, scholl_init

include("running_stats.jl")

function gen_trial(wm::SchollWM, targets,
                   metrics::Metrics,
                   rejuv_steps::Int64 = 100,
                   particles = 10)
    nsteps = size(targets, 2)
    pf = initialize_particle_filter(peak_chain, (0, wm, metrics),
                                    choicemap(), particles)
    obj = 1
    for i = 1:nsteps
        obs = choicemap((:states => i => :metrics,  targets[:, i]))
        particle_filter_step!(pf,
                              (i, wm, metrics),
                              (UnknownChange(), NoChange(), NoChange()),
                              obs)
        maybe_resample!(pf)
        obj = categorical(Fill(1.0 / 3, 3))
        Threads.@threads for p = 1:particles
            for _ = 1:rejuv_steps
                new_tr, w = regenerate(pf.traces[p],
                                    Gen.select(:states => i => :deltas => obj))
                if log(rand()) < w
                    pf.traces[p] = new_tr
                    pf.log_weights[p] += w
                end
            end
        end
    end
    # Extract MAP
    best_idx = argmax(get_log_weights(pf))
    best_tr = pf.traces[best_idx]
    (_, sofar) = get_retval(best_tr)
    nmetrics = length(metrics.funcs)
    vals = Vector{SVector{nmetrics, Float64}}(undef, nsteps)
    for i = 1:nsteps
        vals[i] = SVector{nmetrics, Float64}(metrics((sofar[i],)))
    end

    gorilla = Dict(
        # onset time
        :frame =>
            uniform_discrete(round(Int64, 0.4 * nsteps),
                             round(Int64, 0.6 * nsteps)),
        :parent =>
            uniform_discrete(1, wm.n_dots),
        # speed to move across
        :speedx => normal(wm.vel, 1.0),
        :speedy => normal(wm.vel, 1.0),
    )
    trial = Dict(
        :positions => map(extract_positions, sofar),
        :gorilla =>  gorilla,
    )
    return trial, vals
end

function extract_positions(state::SchollState)
    objects = state.objects
    no = length(objects)
    step = Vector{S2V}(undef, no)
    for k in 1:no
        obj = objects[k]
        step[k] = obj.pos
    end
    return step
end

@gen static function peak_kernel(t::Int, prev::SchollState,
                                 wm::SchollWM, metrics::Metrics)
    deltas ~ Gen.Map(scholl_delta)(Fill(wm, wm.n_dots))
    next::SchollState = MOTCore.step(wm, prev, deltas)
    mus = metrics((next,))
    metrics ~ broadcasted_normal(mus, 1.0)
    return next
end

@gen static function peak_chain(k::Int, wm::SchollWM, metrics::Metrics)
    init_state ~ scholl_init(wm)
    states ~ Gen.Unfold(peak_kernel)(k, init_state, wm, metrics)
    result = (init_state, states)
    return result
end

# Assumes length(targets) > 1
function target_coherence(st::SchollState, targets::Vector{Int64},
                          w::Float64 = 1.0)
    dist_pos = 0.0
    dist_vel = 0.0
    n = length(targets)
    c = 0
    @inbounds for i = 1:(n-1)
        obj_a = st.objects[i]
        pos_a = get_pos(obj_a)
        vel_a = get_vel(obj_a)
        for j = (i+1):n
            obj_b = st.objects[j]
            pos_b = get_pos(obj_b)
            vel_b = get_vel(obj_b)

            dist_pos += norm(pos_a - pos_b)
            dist_vel += angle_between(vel_a, vel_b)
            c += 1
        end
    end
    (dist_pos + w * dist_vel) / c
end

angle_between(a::S2V, b::S2V) = atand(norm(cross(a,b)),dot(a,b))

function target_ecc(st::SchollState, targets::Vector{Int64})
    n = length(targets)
    ecc = 0.0
    @inbounds for i = 1:n
        obj = st.objects[i]
        pos = get_pos(obj)
        ecc = max(ecc, norm(pos))
    end
    ecc
end

function metric_target(f::Function, target::Float64, args...)
    function clamped(st::SchollState)
        y = f(st, args...)
        return abs(y - target)
    end
    return clamped
end

function target_speed(st::SchollState, target::Int64)
    norm(get_vel(st.objects[target]))
end

function main()

    dataset = "target_ensemble"
    version = "0.1"
    nscenes = 6
    nexamples = 2
    fps = 24
    duration = 10 # seconds
    frames = fps * duration
    data_dir = "/spaths/datasets/$(dataset)"
    isdir(data_dir) || mkdir(data_dir)
    out_dir = "$(data_dir)/$(version)"
    isdir(out_dir) || mkdir(out_dir)
    out_path = "$(out_dir)/dataset.json"

    wm = SchollWM(
        ;
        n_dots = 8,
        dot_radius = 5.0,
        area_width = 720.0,
        area_height = 480.0,
        vel=4.5,
        vel_min = 3.0,
        vel_max = 6.0,
        vel_step = 0.8,
        vel_prob = 0.50
    )

    # dataset parameters
    metrics = Metrics(
        [
            x -> target_coherence(x, [1,2,3], 0.01),
            x -> target_speed(x, 4),
            x -> target_ecc(x, [1,2,3])
        ],
        [:tco, :tvl, :tec]
    )
    nm = length(metrics.funcs)

    @time stats = warmup(wm, metrics, frames, 1000)
    display(stats)

    tco_mu, tco_sd = stats[:tco]
    tvl_mu, tvl_sd = stats[:tvl]
    tec_mu, tec_sd = stats[:tec]

    frame_targets = [
        tco_mu - 3*tco_sd;
        1.5*wm.vel;
        0.25 * wm.area_height;
    ]
    targets = repeat(frame_targets, inner = (1, frames))

    trials = []
    for i = 1:nscenes
        trial, vals =
            gen_trial(wm, targets, metrics, 30, 100)
        push!(trials, trial)

        for (mi, m) = enumerate(metrics.names)
            vs = map(x -> x[mi], vals)
            plt = lineplot(1:frames,
                           targets[mi, :],
                           title = String(m),
                           xlabel = "time",
                           name = "target",
                           )
            lineplot!(plt, 1:frames, vs, name = "measured")
            display(plt)
        end
    end
    data = Dict()
    data[:trials] = trials
    data[:manifest] = Dict(:ntrials => nscenes,
                           :duration => duration,
                           :fps => fps,
                           :frames => frames)
    open("$(out_dir)/dataset.json", "w") do io
        JSON3.write(io, data)
    end
    trials = []
    for i = 1:nexamples
        trial, vals =
            gen_trial(wm, targets, metrics, 30, 100)
        push!(trials, trial)
    end
    data = Dict()
    data[:trials] = trials
    data[:manifest] = Dict(:ntrials => nexamples,
                           :duration => duration,
                           :fps => fps,
                           :frames => frames)
    open("$(out_dir)/examples.json", "w") do io
        JSON3.write(io, data)
    end
    cp(@__FILE__, "$(out_dir)/script.jl"; force = true)
end

main();
